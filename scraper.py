"""Agentic web-research RAG pipeline.

Extracted from WebScrapper_Agents.ipynb and wrapped in a reusable, configurable
``ResearchAgent`` so it can be driven from a Streamlit UI (or anywhere else).

Pipeline (LangGraph):
    search -> playwright -> parse -> chunk -> embed -> faiss -> retrieve
    -> evaluate -> (research -> search ...loop)  OR  answer -> verify -> END

Accuracy/trust design: chunks carry provenance (url + domain); retrieval uses
cosine (normalized IndexFlatIP) with an abstention floor and a cross-encoder
reranker; generation is deterministic (temperature=0), cites its sources with
[n] markers, and emits a machine-readable claims/confidence/abstained object; a
verify pass checks each claim against its cited chunks and counts cross-domain
corroboration. The goal is cited, confidence-scored, abstaining answers — not a
correctness oracle.
"""

from __future__ import annotations

import io
import re
import json
import time
import queue
import asyncio
import threading
from dataclasses import dataclass
from urllib.parse import urlsplit
from typing import Iterator, Optional, Tuple, TypedDict

import httpx
import numpy as np
import faiss
from ddgs import DDGS
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq, InternalServerError, RateLimitError
from langgraph.graph import StateGraph, START, END


SERPER_DOC_URL = "Google instant answer (Serper)"  # synthetic, never a real domain


# --------------------------------------------------------------------------- #
# Configuration & state
# --------------------------------------------------------------------------- #
@dataclass
class ScraperConfig:
    """All knobs the pipeline exposes to the UI."""

    model: str = "llama-3.3-70b-versatile"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_searches: int = 3
    max_results: int = 10
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 5
    first_stage_k: int = 20          # candidates fetched before reranking
    min_similarity: float = 0.30     # cosine floor (tunable ~0.25-0.35 for MiniLM)
    rerank_min_score: float = 0.0    # ms-marco LOGIT floor (not a probability)
    page_timeout_ms: int = 15000
    pdf_max_pages: int = 50          # cap pages read per PDF (keeps embedding fast)
    pdf_max_chars: int = 400_000     # hard cap on extracted text per PDF
    serper_api_key: str = ""         # if set, use Serper (Google); else DuckDuckGo


class State(TypedDict, total=False):
    query: str
    original_query: str
    urls: list
    search_provider: str
    serper_answer: str
    html_pages: list
    pdf_docs: list
    documents: list
    all_documents: list              # evidence accumulated across search rounds
    chunks: list                     # list[dict{text, url, domain}]
    embeddings: object
    faiss_index: object
    retrieved_chunks: list           # list[dict{text, url, domain, sim, rerank?}]
    top_rerank_score: float
    answer: str
    claims: list                     # list[dict{text, sources:[int], supported:bool}]
    confidence: str                  # 'high' | 'medium' | 'low' (model self-report)
    abstained: bool
    confidence_score: float          # 0-1, computed by verify_node
    corroboration_max: int           # max distinct independent domains for any claim
    unverified_claims: list
    conflicts: list
    search_count: int
    enough_info: bool


# Friendly labels for each graph node, surfaced as progress in the UI.
STEP_LABELS = {
    "search": "🔍 Searching the web",
    "playwright": "🌐 Scraping pages (HTML + PDFs)",
    "parse": "📄 Extracting text",
    "chunk": "✂️ Chunking documents",
    "embed": "🧮 Computing embeddings",
    "faiss": "🗂️ Building vector index",
    "retrieve": "📥 Retrieving + reranking chunks",
    "evaluate": "🤔 Evaluating whether the context is sufficient",
    "research": "♻️ Refining the search query",
    "answer": "✍️ Writing the cited answer",
    "verify": "✅ Verifying claims against sources",
}


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class ResearchAgent:
    """Wraps the compiled LangGraph together with its LLM, embedder + reranker."""

    def __init__(
        self,
        client: Groq,
        embedding_model: SentenceTransformer,
        config: Optional[ScraperConfig] = None,
        reranker=None,  # sentence_transformers.CrossEncoder | None
    ):
        self.client = client
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.config = config or ScraperConfig()
        self.graph = self._build_graph()

    # ----- LLM helper ----------------------------------------------------- #
    def generate(
        self,
        prompt: str,
        retries: int = 5,
        base_delay: int = 2,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
    ) -> str:
        kwargs = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": 1,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except (InternalServerError, RateLimitError):
                if attempt == retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))
        return ""

    def _generate_json(self, prompt: str, max_tokens: int = 1024) -> Tuple[dict, str]:
        """Deterministic JSON generation, resilient to models without json mode."""
        try:
            raw = self.generate(prompt, temperature=0.0, max_tokens=max_tokens,
                                 response_format={"type": "json_object"})
        except Exception:
            raw = self.generate(prompt, temperature=0.0, max_tokens=max_tokens)
        return self._loads_lenient(raw), raw

    @staticmethod
    def _loads_lenient(raw: str) -> dict:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
        return {}

    @staticmethod
    def _domain(url: str) -> str:
        """Registrable domain (so www.x.com and sub.x.com count as one source)."""
        net = urlsplit(url).netloc.lower()
        if net.startswith("www."):
            net = net[4:]
        return ".".join(net.split(".")[-2:]) if net else url

    # ----- Search --------------------------------------------------------- #
    def _serper_search(self, query: str) -> dict:
        """Raw Google results from the Serper.dev JSON API."""
        resp = httpx.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": self.config.serper_api_key,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": self.config.max_results},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _serper_answer(data: dict) -> str:
        """Build Google's direct-answer text from Serper's answerBox /
        knowledgeGraph / aiOverview fields (whichever are present)."""
        parts = []

        ab = data.get("answerBox") or {}
        ans = ab.get("answer") or ab.get("snippet") or ""
        if ans:
            parts.append(f"Google answer box — {ab.get('title', '')}: {ans}".strip(" —:"))

        kg = data.get("knowledgeGraph") or {}
        if kg:
            attrs = "; ".join(f"{k}: {v}" for k, v in (kg.get("attributes") or {}).items())
            kg_text = " ".join(x for x in [kg.get("title", ""), kg.get("description", ""), attrs] if x).strip()
            if kg_text:
                parts.append(f"Google knowledge graph — {kg_text}")

        ai = data.get("aiOverview")  # Serper's (partial) AI Overview, when returned
        if isinstance(ai, dict):
            ai = ai.get("text") or ai.get("snippet") or ""
        if isinstance(ai, str) and ai.strip():
            parts.append(f"Google AI overview — {ai.strip()}")

        return "\n".join(parts)

    def _ddg_search(self, query: str) -> list:
        """Fallback: DuckDuckGo via the ddgs package (no API key)."""
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=self.config.max_results))
        return [r.get("href") or r.get("url") for r in results if r.get("href") or r.get("url")]

    def search_node(self, state: State) -> State:
        query = state["query"]
        urls, provider = [], ""

        if self.config.serper_api_key:
            try:
                data = self._serper_search(query)
                urls = [i.get("link") for i in data.get("organic", []) if i.get("link")]
                urls = urls[: self.config.max_results]
                answer = self._serper_answer(data)
                if answer:  # keep the last non-empty instant answer
                    state["serper_answer"] = answer
                provider = "Serper (Google)"
            except Exception:
                urls, provider = [], "DuckDuckGo (Serper failed)"

        if not urls:  # no key, or Serper returned nothing / errored
            try:
                urls = self._ddg_search(query)
            except Exception:
                urls = []
            provider = provider or "DuckDuckGo"

        state["urls"] = urls
        state["search_provider"] = provider
        return state

    # ----- Scrape (HTML + PDF) ------------------------------------------- #
    @staticmethod
    def _is_pdf_url(url: str) -> bool:
        return urlsplit(url).path.lower().endswith(".pdf")

    def _pdf_bytes_to_text(self, data: bytes) -> str:
        """Extract text from PDF bytes, capped by config to keep things fast."""
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # surfaced clearly instead of a cryptic failure
            raise ImportError("Reading PDFs requires 'pypdf' (pip install pypdf).") from exc

        reader = PdfReader(io.BytesIO(data))
        parts, total = [], 0
        for page in reader.pages[: self.config.pdf_max_pages]:
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            parts.append(text)
            total += len(text)
            if total >= self.config.pdf_max_chars:
                break
        return "\n".join(parts)

    async def _fetch_pdf_text(self, http: httpx.AsyncClient, url: str) -> str:
        try:
            resp = await http.get(url)
            resp.raise_for_status()
            data = resp.content
            if not data[:5].startswith(b"%PDF"):  # not actually a PDF
                return ""
            return self._pdf_bytes_to_text(data)
        except Exception:
            return ""

    async def playwright_node(self, state: State) -> State:
        html_pages: list = []
        pdf_docs: list = []
        urls = state.get("urls") or []
        if urls:
            timeout_s = self.config.page_timeout_ms / 1000
            headers = {"User-Agent": "Mozilla/5.0 (research-assistant)"}
            async with async_playwright() as p, httpx.AsyncClient(
                follow_redirects=True, timeout=timeout_s, headers=headers
            ) as http:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                for url in urls:
                    # PDFs can't be read via the browser DOM — fetch + extract directly.
                    if self._is_pdf_url(url):
                        text = await self._fetch_pdf_text(http, url)
                        if text.strip():
                            pdf_docs.append({"url": url, "content": text})
                        continue
                    try:
                        await page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=self.config.page_timeout_ms,
                        )
                        html_pages.append({"url": url, "html": await page.content()})
                    except Exception:
                        continue
                await browser.close()
        state["html_pages"] = html_pages
        state["pdf_docs"] = pdf_docs
        return state

    # ----- Parse / chunk / embed / index --------------------------------- #
    def parse_node(self, state: State) -> State:
        # Accumulate evidence across rounds so the loop is additive, not amnesiac.
        acc = state.get("all_documents") or []
        seen = {d["url"] for d in acc}

        def add(url: str, content: str) -> None:
            if url and url not in seen and content and content.strip():
                acc.append({"url": url, "content": content})
                seen.add(url)

        answer = state.get("serper_answer")
        if answer:  # Google's direct answer — a top-priority source
            add(SERPER_DOC_URL, answer)
        for page in state.get("html_pages", []):
            soup = BeautifulSoup(page["html"], "html.parser")
            add(page["url"], soup.get_text(separator=" ", strip=True))
        for doc in state.get("pdf_docs", []):
            add(doc["url"], doc.get("content", ""))

        state["all_documents"] = acc
        state["documents"] = acc
        return state

    def chunk_node(self, state: State) -> State:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            keep_separator=True,
        )
        seen, chunks = set(), []
        for doc in state.get("documents", []):
            for piece in splitter.split_text(doc["content"]):
                key = piece.strip()
                if not key or key in seen:  # drop blanks + exact mirrored dups
                    continue
                seen.add(key)
                chunks.append({"text": piece, "url": doc["url"], "domain": self._domain(doc["url"])})
        state["chunks"] = chunks
        return state

    def embedding_node(self, state: State) -> State:
        chunks = state.get("chunks") or []
        if not chunks:
            state["embeddings"] = []
            return state
        state["embeddings"] = self.embedding_model.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,  # unit vectors -> inner product == cosine
            batch_size=64,
            convert_to_numpy=True,
        )
        return state

    def faiss_node(self, state: State) -> State:
        embeddings = state.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            state["faiss_index"] = None
            return state
        vectors = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])  # cosine on the unit vectors above
        index.add(vectors)
        state["faiss_index"] = index
        return state

    def retrieve_node(self, state: State) -> State:
        chunks = state.get("chunks") or []
        if not state.get("faiss_index") or not chunks:
            state["retrieved_chunks"] = []
            state["top_rerank_score"] = float("-inf")
            return state

        # Use the ORIGINAL query so entity/number tokens survive re-query drift.
        query = state.get("original_query") or state["query"]
        qv = self.embedding_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        k = min(self.config.first_stage_k, len(chunks))
        scores, indices = state["faiss_index"].search(qv, k)
        candidates = [
            {**chunks[i], "sim": float(s)}
            for s, i in zip(scores[0], indices[0])
            if 0 <= i < len(chunks)
        ]
        # Cosine abstention floor: weak matches don't count as evidence.
        candidates = [c for c in candidates if c["sim"] >= self.config.min_similarity]

        if self.reranker and candidates:
            pairs = [[query, c["text"]] for c in candidates]
            rr = self.reranker.predict(pairs)  # ONE batched cross-encoder call
            for c, s in zip(candidates, rr):
                c["rerank"] = float(s)
            candidates.sort(key=lambda c: c["rerank"], reverse=True)
            candidates = [c for c in candidates if c["rerank"] >= self.config.rerank_min_score]
            state["top_rerank_score"] = candidates[0]["rerank"] if candidates else float("-inf")
        else:
            candidates.sort(key=lambda c: c["sim"], reverse=True)
            state["top_rerank_score"] = float("inf")  # no rerank gate without a reranker

        state["retrieved_chunks"] = candidates[: self.config.top_k]
        return state

    # ----- Evaluate / route / re-query ----------------------------------- #
    def evaluate_node(self, state: State) -> State:
        if state.get("search_count", 0) >= self.config.max_searches:
            state["enough_info"] = True
            return state

        retrieved = state.get("retrieved_chunks") or []
        if not retrieved:
            state["enough_info"] = False
            return state
        # The LLM can hallucinate YES on thin context — require decent rerank too.
        if state.get("top_rerank_score", float("-inf")) < self.config.rerank_min_score:
            state["enough_info"] = False
            return state

        context = "\n".join(c["text"] for c in retrieved)
        prompt = f"""Question:
{state.get('original_query', state['query'])}

Context:
{context}

Can the question be answered confidently from this context?
Reply only YES or NO."""
        state["enough_info"] = "YES" in self.generate(prompt, temperature=0.0, max_tokens=5).strip().upper()
        return state

    @staticmethod
    def route_node(state: State) -> str:
        return "answer" if state.get("enough_info") else "research"

    def research_node(self, state: State) -> State:
        retrieved = state.get("retrieved_chunks") or []
        context = "\n".join(c["text"] for c in retrieved)[:1500]
        original = state.get("original_query", state["query"])
        prompt = f"""The previous search did not fully answer the question.
Question: {original}
What we found so far:
{context}

Write ONE improved web search query targeting the MISSING information.
Output only the query."""
        # temperature>0 so a re-query of an already-failed query actually diverges.
        new_query = self.generate(prompt, temperature=0.4, max_tokens=60).strip().strip('"')
        state["query"] = new_query or original
        state["search_count"] = state.get("search_count", 0) + 1
        return state

    # ----- Answer (cited, structured) ------------------------------------ #
    def answer_node(self, state: State) -> State:
        retrieved = state.get("retrieved_chunks") or []
        question = state.get("original_query") or state["query"]

        if not retrieved:  # floor abstain — nothing cleared the evidence bar
            state["answer"] = "I could not find sufficient information in the retrieved documents."
            state["claims"] = []
            state["confidence"] = "low"
            state["abstained"] = True
            return state

        blocks = [
            f'[{i}] ({c.get("domain", "")}) {c.get("url", "")}\n{c["text"]}'
            for i, c in enumerate(retrieved, 1)
        ]
        context = "\n\n".join(blocks)
        prompt = f"""You are a research assistant. Use ONLY the numbered sources below.
Rules:
(1) Every factual sentence MUST end with citation markers like [1] or [2][3] referencing the SOURCE numbers you used.
(2) Do NOT use outside knowledge.
(3) If the sources do not contain enough to answer, set "abstained" to true and set "answer" to exactly: "I could not find sufficient information in the retrieved documents."

Return ONLY JSON with this shape:
{{"answer": "<text with [n] citations>", "claims": [{{"text": "<one factual sentence>", "sources": [1], "supported": true}}], "confidence": "high|medium|low", "abstained": false}}

SOURCES:
{context}

QUESTION:
{question}"""
        parsed, raw = self._generate_json(prompt)
        if not parsed.get("answer"):  # degraded parse — keep the prose, no structure
            parsed = {"answer": (raw or "").strip(), "claims": [], "confidence": "low", "abstained": False}

        n = len(retrieved)
        claims = []
        for cl in parsed.get("claims") or []:
            if not isinstance(cl, dict):
                continue
            srcs = [s for s in (cl.get("sources") or []) if isinstance(s, int) and 1 <= s <= n]
            claims.append({"text": cl.get("text", ""), "sources": srcs, "supported": bool(cl.get("supported", False))})

        conf = parsed.get("confidence")
        state["answer"] = parsed.get("answer") or ""
        state["claims"] = claims
        state["confidence"] = conf if conf in ("high", "medium", "low") else "low"
        state["abstained"] = bool(parsed.get("abstained", False))
        return state

    # ----- Verify (grounding + corroboration + conflicts) ---------------- #
    _SCORE = {"SUPPORTED": 1.0, "SINGLE_SOURCE": 0.6, "UNVERIFIED": 0.15, "CONFLICTING": 0.0}

    def verify_node(self, state: State) -> State:
        retrieved = state.get("retrieved_chunks") or []
        claims = state.get("claims") or []
        state["conflicts"] = []
        state["unverified_claims"] = []
        state["corroboration_max"] = 0

        if state.get("abstained") or not claims or not retrieved:
            state["confidence_score"] = 0.0 if state.get("abstained") else 0.15
            return state

        # ONE batched grounding call: judge each claim ONLY against its cited chunks.
        items = []
        for idx, cl in enumerate(claims):
            cited = [retrieved[s - 1] for s in cl.get("sources", []) if 1 <= s <= len(retrieved)]
            ev = "\n".join(f"    - {c['text']}" for c in cited) or "    - (no sources cited)"
            items.append(f"[{idx}] CLAIM: {cl['text']}\n  EVIDENCE:\n{ev}")
        prompt = (
            "You are a strict fact-checker. For each numbered claim, decide ONLY from its EVIDENCE:\n"
            "- supported: true if the evidence directly states or clearly implies the claim.\n"
            "- conflicting: true ONLY if the evidence snippets disagree on the SAME quantity "
            "(same metric, unit, AND time period); never flag different years/metrics as conflicting.\n\n"
            'Return ONLY JSON: {"verdicts":[{"i":0,"supported":true,"conflicting":false}]}\n\n'
            "CLAIMS:\n" + "\n\n".join(items)
        )
        parsed, _ = self._generate_json(prompt)
        verd = {}
        for v in parsed.get("verdicts") or []:
            if isinstance(v, dict) and isinstance(v.get("i"), int):
                verd[v["i"]] = (bool(v.get("supported")), bool(v.get("conflicting")))

        statuses, unverified, conflicts, corr_max = [], [], [], 0
        for idx, cl in enumerate(claims):
            cited = [retrieved[s - 1] for s in cl.get("sources", []) if 1 <= s <= len(retrieved)]
            # distinct INDEPENDENT domains (the Serper aggregator never corroborates alone)
            domains = {c["domain"] for c in cited if c.get("url") != SERPER_DOC_URL and c.get("domain")}
            corr_max = max(corr_max, len(domains))
            supported, conflicting = verd.get(idx, (False, False))

            if conflicting:
                status = "CONFLICTING"
                conflicts.append(cl["text"])
            elif not cited or not supported:
                status = "UNVERIFIED"
            elif len(domains) >= 2:
                status = "SUPPORTED"
            elif len(domains) == 1:
                status = "SINGLE_SOURCE"
            else:
                status = "UNVERIFIED"

            cl["status"] = status
            statuses.append(status)
            if status in ("UNVERIFIED", "CONFLICTING"):
                unverified.append(cl["text"])

        score = sum(self._SCORE[s] for s in statuses) / len(statuses)
        # Mildly temper by mean retrieval strength (sim in [-1,1] -> [0,1]).
        sims = [c.get("sim", 0.0) for c in retrieved]
        mean_sim = max(0.0, min(1.0, (sum(sims) / len(sims)) if sims else 0.0))
        score *= (0.6 + 0.4 * mean_sim)

        state["confidence_score"] = round(score, 3)
        state["corroboration_max"] = corr_max
        state["unverified_claims"] = unverified
        state["conflicts"] = conflicts

        if unverified:
            note = f"\n\n_[Unverified] {len(unverified)} claim(s) could not be confirmed from the cited sources._"
            state["answer"] = (state.get("answer") or "") + note
            state["confidence"] = "low"
        return state

    # ----- Graph wiring --------------------------------------------------- #
    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("search", self.search_node)
        graph.add_node("playwright", self.playwright_node)
        graph.add_node("parse", self.parse_node)
        graph.add_node("chunk", self.chunk_node)
        graph.add_node("embed", self.embedding_node)
        graph.add_node("faiss", self.faiss_node)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("evaluate", self.evaluate_node)
        graph.add_node("research", self.research_node)
        graph.add_node("answer", self.answer_node)
        graph.add_node("verify", self.verify_node)

        graph.add_edge(START, "search")
        graph.add_edge("search", "playwright")
        graph.add_edge("playwright", "parse")
        graph.add_edge("parse", "chunk")
        graph.add_edge("chunk", "embed")
        graph.add_edge("embed", "faiss")
        graph.add_edge("faiss", "retrieve")
        graph.add_edge("retrieve", "evaluate")
        graph.add_conditional_edges(
            "evaluate", self.route_node, {"answer": "answer", "research": "research"}
        )
        graph.add_edge("research", "search")
        graph.add_edge("answer", "verify")
        graph.add_edge("verify", END)

        return graph.compile()

    # ----- Public runners ------------------------------------------------- #
    def stream_research(self, query: str) -> Iterator[Tuple[str, dict]]:
        """Run the graph in a background thread, yielding (kind, payload) events.

        kind == "update": payload is {node_name: full_state_after_node}
        kind == "error":  payload is the raised Exception
        """
        events: "queue.Queue" = queue.Queue()
        initial: State = {"query": query, "original_query": query, "search_count": 0}

        def worker():
            async def go():
                try:
                    async for update in self.graph.astream(initial):
                        events.put(("update", update))
                except Exception as exc:  # surfaced to the UI
                    events.put(("error", exc))
                finally:
                    events.put(("done", None))

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(go())
            finally:
                loop.close()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            kind, payload = events.get()
            if kind == "done":
                break
            yield kind, payload
        thread.join()

    def research(self, query: str) -> dict:
        """Run the pipeline to completion and return the merged final state."""
        merged: dict = {}
        for kind, payload in self.stream_research(query):
            if kind == "error":
                raise payload
            merged.update(next(iter(payload.values())))
        return merged
