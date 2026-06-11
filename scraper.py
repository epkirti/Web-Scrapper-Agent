"""Agentic web-research RAG pipeline.

Extracted from WebScrapper_Agents.ipynb and wrapped in a reusable, configurable
``ResearchAgent`` so it can be driven from a Streamlit UI (or anywhere else).

Pipeline (LangGraph):
    search -> playwright -> parse -> chunk -> embed -> faiss -> retrieve
    -> evaluate -> (research -> search ...loop)  OR  answer -> END
"""

from __future__ import annotations

import io
import time
import queue
import asyncio
import threading
from dataclasses import dataclass
from urllib.parse import urlsplit
from typing import Callable, Iterator, Optional, Tuple, TypedDict

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


# --------------------------------------------------------------------------- #
# Configuration & state
# --------------------------------------------------------------------------- #
@dataclass
class ScraperConfig:
    """All knobs the pipeline exposes to the UI."""

    model: str = "llama-3.3-70b-versatile"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    max_searches: int = 5
    max_results: int = 10
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5
    page_timeout_ms: int = 15000
    pdf_max_pages: int = 50          # cap pages read per PDF (keeps embedding fast)
    pdf_max_chars: int = 400_000     # hard cap on extracted text per PDF
    serper_api_key: str = ""         # if set, use Serper (Google); else DuckDuckGo


class State(TypedDict, total=False):
    query: str
    original_query: str
    urls: list
    search_provider: str
    html_pages: list
    pdf_docs: list
    documents: list
    chunks: list
    embeddings: list
    faiss_index: object
    retrieved_chunks: list
    answer: str
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
    "retrieve": "📥 Retrieving relevant chunks",
    "evaluate": "🤔 Evaluating whether the context is sufficient",
    "research": "♻️ Refining the search query",
    "answer": "✍️ Writing the answer",
}


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class ResearchAgent:
    """Wraps the compiled LangGraph together with its LLM + embedding model."""

    def __init__(
        self,
        client: Groq,
        embedding_model: SentenceTransformer,
        config: Optional[ScraperConfig] = None,
    ):
        self.client = client
        self.embedding_model = embedding_model
        self.config = config or ScraperConfig()
        self.graph = self._build_graph()

    # ----- LLM helper ----------------------------------------------------- #
    def generate(self, prompt: str, retries: int = 5, base_delay: int = 2) -> str:
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except (InternalServerError, RateLimitError):
                if attempt == retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))
        return ""

    # ----- Graph nodes ---------------------------------------------------- #
    def _serper_search(self, query: str) -> list:
        """Google results via the Serper.dev JSON API."""
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
        data = resp.json()
        urls = [item.get("link") for item in data.get("organic", []) if item.get("link")]
        return urls[: self.config.max_results]

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
                urls = self._serper_search(query)
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

    def parse_node(self, state: State) -> State:
        documents = []
        for page in state.get("html_pages", []):
            soup = BeautifulSoup(page["html"], "html.parser")
            documents.append(
                {"url": page["url"], "content": soup.get_text(separator=" ", strip=True)}
            )
        # PDF documents already carry extracted text — add them as-is.
        for doc in state.get("pdf_docs", []):
            documents.append(doc)
        state["documents"] = documents
        return state

    def chunk_node(self, state: State) -> State:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = []
        for doc in state.get("documents", []):
            chunks.extend(splitter.split_text(doc["content"]))
        state["chunks"] = chunks
        return state

    def embedding_node(self, state: State) -> State:
        chunks = state.get("chunks") or []
        if not chunks:
            state["embeddings"] = []
            return state
        state["embeddings"] = self.embedding_model.encode(chunks)
        return state

    def faiss_node(self, state: State) -> State:
        embeddings = state.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            state["faiss_index"] = None
            return state
        vectors = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        state["faiss_index"] = index
        return state

    def retrieve_node(self, state: State) -> State:
        chunks = state.get("chunks") or []
        if not state.get("faiss_index") or not chunks:
            state["retrieved_chunks"] = []
            return state
        query_vector = np.array(self.embedding_model.encode([state["query"]])).astype("float32")
        k = min(self.config.top_k, len(chunks))
        _, indices = state["faiss_index"].search(query_vector, k)
        state["retrieved_chunks"] = [chunks[idx] for idx in indices[0]]
        return state

    def evaluate_node(self, state: State) -> State:
        if state.get("search_count", 0) >= self.config.max_searches:
            state["enough_info"] = True
            return state

        retrieved = state.get("retrieved_chunks") or []
        if not retrieved:
            state["enough_info"] = False
            return state

        context = "\n".join(retrieved)
        prompt = f"""Question:
{state['query']}

Context:
{context}

Can the question be answered confidently from this context?
Reply only YES or NO."""
        state["enough_info"] = "YES" in self.generate(prompt).strip().upper()
        return state

    @staticmethod
    def route_node(state: State) -> str:
        return "answer" if state.get("enough_info") else "research"

    def research_node(self, state: State) -> State:
        prompt = f"""Create a better search query.

Question:
{state['query']}"""
        state["query"] = self.generate(prompt).strip()
        state["search_count"] = state.get("search_count", 0) + 1
        return state

    def answer_node(self, state: State) -> State:
        retrieved = state.get("retrieved_chunks") or []
        context = "\n\n".join(retrieved)
        prompt = f"""You are a helpful research assistant.

Answer the question ONLY using the provided context.
If the answer is not present in the context, say:
"I could not find sufficient information in the retrieved documents."

Context:
{context}

Question:
{state['query']}"""
        state["answer"] = self.generate(prompt)
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
        graph.add_edge("answer", END)

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
