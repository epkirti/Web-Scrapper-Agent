"""Accuracy-first agentic web-research RAG pipeline.

Drop-in upgrade of the original ``scraper.py``. Same public surface
(``ScraperConfig``, ``ResearchAgent``, ``stream_research``, ``STEP_LABELS``)
but rebuilt around one goal: *when the agent states a fact, that fact must be
traceable to evidence it actually scraped* — and when it can't verify, it says
so instead of guessing.

What changed vs. the original pipeline (and why it matters for accuracy):

1.  Query planning      — the LLM expands the question into several diverse
                          search queries per round and targets known gaps; the
                          user's original question is never overwritten.
2.  Search              — Serper (Google) with DuckDuckGo fallback; result
                          snippets + answer box are kept as cited mini-evidence;
                          URLs are deduped, tracking params stripped, low-value
                          social domains skipped, authoritative domains boosted,
                          and the scrape budget is spread across domains.
3.  Scraping            — fast parallel httpx first, Playwright rendering only
                          as a fallback for JS-heavy pages; PDFs via pypdf.
4.  Extraction          — Trafilatura main-content extraction (tables kept)
                          instead of raw ``soup.get_text`` boilerplate soup.
5.  Chunking            — sentence-aware splitting; every chunk carries its
                          source URL / title / date / domain.
6.  Retrieval           — hybrid: dense (FAISS, cosine) + lexical (BM25),
                          fused with Reciprocal Rank Fusion across multiple
                          query phrasings, then re-scored by a cross-encoder
                          reranker, with a per-domain diversity cap.
7.  Answering           — strict grounded prompt: per-sentence [n] citations,
                          numbers copied verbatim with units + time period,
                          multi-source corroboration preferred, answer-box
                          treated as a hint that needs corroboration.
8.  Verification        — an adversarial fact-check pass labels unsupported
                          claims; if found, the agent either searches again to
                          close the gap or rewrites the answer keeping only
                          supported claims, and reports a confidence grade.

Pipeline (LangGraph):
    plan -> search -> scrape -> index -> retrieve -> evaluate
        -> (loop to plan)  OR  answer -> verify -> (loop to plan | finalize)
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from typing import Iterator, List, Optional, Tuple, TypedDict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import numpy as np
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

try:  # optional but strongly recommended
    import faiss
except Exception:  # pragma: no cover
    faiss = None

try:  # optional but strongly recommended — clean main-content extraction
    import trafilatura
except Exception:  # pragma: no cover
    trafilatura = None

try:  # optional but strongly recommended — lexical half of hybrid retrieval
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None


# --------------------------------------------------------------------------- #
# Configuration & state
# --------------------------------------------------------------------------- #
@dataclass
class ScraperConfig:
    """All knobs the pipeline exposes to the UI."""

    model: str = "llama-3.3-70b-versatile"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    max_searches: int = 4              # max research rounds (plan->search loops)
    max_results: int = 10              # search results requested per query
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 6                     # final evidence chunks given to the LLM
    page_timeout_ms: int = 15000
    pdf_max_pages: int = 50
    pdf_max_chars: int = 400_000
    serper_api_key: str = ""           # if set, use Serper (Google); else DuckDuckGo

    # ----- accuracy knobs (new) ------------------------------------------- #
    queries_per_round: int = 3         # diverse search queries generated per round
    max_pages_per_round: int = 8       # scrape budget per round
    retrieve_k: int = 24               # hybrid candidates fed to the reranker
    rerank: bool = True                # cross-encoder reranking on/off
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    verify_answer: bool = True         # adversarial fact-check pass on/off
    min_extract_chars: int = 300       # below this, retry the page via Playwright
    per_doc_char_cap: int = 60_000     # hard cap per scraped document
    max_total_chunks: int = 1500       # index size cap (keeps CPU embedding fast)
    max_chunks_per_domain: int = 3     # diversity cap in the final evidence set
    use_search_snippets: bool = True   # index search snippets as cited mini-docs
    playwright_fallback: bool = True   # render JS-heavy pages when httpx fails
    temperature: float = 0.1           # low temp -> fewer creative liberties


class State(TypedDict, total=False):
    question: str            # the user's original question — never overwritten
    query: str               # current lead query (for UI display)
    queries: List[str]       # all queries for the current round
    tried_queries: List[str]
    missing: str             # what the auditor/verifier says is still missing
    search_round: int
    urls: List[str]          # all URLs selected for scraping so far
    new_urls: List[str]
    search_provider: str
    serper_answer: str
    snippets: List[dict]
    documents: List[dict]
    n_documents: int
    n_chunks: int
    retrieved: List[dict]
    retrieved_chunks: List[str]
    sources: List[dict]
    enough_info: bool
    answer: str
    verification: dict
    confidence: str


# Friendly labels for each graph node, surfaced as progress in the UI.
STEP_LABELS = {
    "plan": "🧠 Planning search queries",
    "search": "🔍 Searching the web",
    "scrape": "🌐 Reading pages (httpx → Playwright fallback)",
    "index": "📚 Chunking + hybrid index (FAISS + BM25)",
    "retrieve": "🎯 Retrieving evidence (hybrid + rerank)",
    "evaluate": "⚖️ Auditing evidence sufficiency",
    "answer": "✍️ Drafting cited answer",
    "verify": "🔬 Fact-checking the draft",
    "finalize": "✅ Finalizing",
}


# --------------------------------------------------------------------------- #
# Constants & small helpers (module-level so they're easy to test / patch)
# --------------------------------------------------------------------------- #
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.7",
}

# Social / video / Q&A domains: JS-walled, login-walled or too noisy to be
# worth the scrape budget. Their *snippets* still flow in via search results.
BLOCKED_DOMAINS = {
    "x.com", "twitter.com", "facebook.com", "instagram.com", "linkedin.com",
    "pinterest.com", "tiktok.com", "youtube.com", "youtu.be", "quora.com",
    "reddit.com", "threads.net",
}

# Substring hints that a domain is an authoritative / primary source.
AUTHORITY_HINTS = (
    ".gov", ".nic.in", ".edu", ".ac.in", ".ac.uk", ".int", "wikipedia.org",
    "who.int", "worldbank.org", "imf.org", "un.org", "oecd.org", "rbi.org.in",
    "pib.gov.in", "censusindia", "mospi", "data.gov", "indiabudget",
    "europa.eu", "reuters.com", "apnews.com", "nature.com", "nih.gov",
    "arxiv.org", "sciencedirect.com",
)

_TRACKING_PREFIXES = ("utm_", "mc_")
_TRACKING_PARAMS = {"fbclid", "gclid", "igshid", "ref", "ref_src"}

RRF_K = 60  # standard Reciprocal Rank Fusion constant


def canonical_url(url: str) -> str:
    """Normalise a URL so duplicates collapse (drop fragment + tracking params)."""
    try:
        parts = urlsplit(url.strip())
        query = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k not in _TRACKING_PARAMS and not k.lower().startswith(_TRACKING_PREFIXES)
        ]
        return urlunsplit(
            (parts.scheme.lower(), parts.netloc.lower(), parts.path, urlencode(query), "")
        )
    except Exception:
        return url


def domain_of(url: str) -> str:
    try:
        host = urlsplit(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def is_authority(domain: str) -> bool:
    d = (domain or "").lower()
    return any(hint in d for hint in AUTHORITY_HINTS)


def is_blocked(domain: str) -> bool:
    d = (domain or "").lower()
    return d in BLOCKED_DOMAINS or any(d.endswith("." + b) for b in BLOCKED_DOMAINS)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def extract_json(text: str) -> dict:
    """Best-effort JSON object extraction from an LLM reply."""
    if not text:
        return {}
    cleaned = re.sub(r"```(?:json)?", "", text).strip("` \n\t")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# ----- search providers ----------------------------------------------------- #
def serper_search(query: str, api_key: str, n: int) -> dict:
    """Raw Google results from the Serper.dev JSON API."""
    resp = httpx.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": n},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def build_serper_answer(data: dict) -> str:
    """Google's direct-answer text (answer box / knowledge graph / AI overview)."""
    parts = []
    ab = data.get("answerBox") or {}
    ans = ab.get("answer") or ab.get("snippet") or ""
    if ans:
        parts.append(f"Google answer box — {ab.get('title', '')}: {ans}".strip(" —:"))
    kg = data.get("knowledgeGraph") or {}
    if kg:
        attrs = "; ".join(f"{k}: {v}" for k, v in (kg.get("attributes") or {}).items())
        kg_text = " ".join(
            x for x in [kg.get("title", ""), kg.get("description", ""), attrs] if x
        ).strip()
        if kg_text:
            parts.append(f"Google knowledge graph — {kg_text}")
    ai = data.get("aiOverview")
    if isinstance(ai, dict):
        ai = ai.get("text") or ai.get("snippet") or ""
    if isinstance(ai, str) and ai.strip():
        parts.append(f"Google AI overview — {ai.strip()}")
    return "\n".join(parts)


def ddg_search(query: str, n: int) -> List[dict]:
    """DuckDuckGo fallback via the ddgs package (no API key needed)."""
    try:
        from ddgs import DDGS
    except ImportError:  # pragma: no cover - legacy package name
        from duckduckgo_search import DDGS
    out: List[dict] = []
    results = DDGS().text(query, max_results=n)
    for item in results or []:
        url = item.get("href") or item.get("url") or item.get("link")
        if not url:
            continue
        out.append(
            {
                "url": url,
                "title": item.get("title", ""),
                "snippet": item.get("body") or item.get("snippet") or "",
                "date": "",
            }
        )
    return out


# ----- extraction ------------------------------------------------------------ #
def extract_main_text(html: str, url: str = "") -> Tuple[str, str, str]:
    """Return (main_text, title, date) from raw HTML.

    Trafilatura strips navigation/footer/cookie boilerplate and keeps tables —
    that alone removes most of the noise that used to drown the embeddings.
    """
    if not html:
        return "", "", ""
    text, title, dt = "", "", ""
    if trafilatura is not None:
        try:
            text = trafilatura.extract(
                html,
                url=url or None,
                include_tables=True,
                include_comments=False,
                favor_recall=True,
            ) or ""
        except Exception:
            text = ""
        try:
            meta = trafilatura.extract_metadata(html)
            if meta:
                title = meta.title or ""
                dt = meta.date or ""
        except Exception:
            pass
    if not text:  # fallback: cleaned BeautifulSoup
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
            if not title and soup.title and soup.title.string:
                title = soup.title.string.strip()
        except Exception:
            text = ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, title.strip(), (dt or "").strip()


def pdf_to_text(data: bytes, max_pages: int, max_chars: int) -> str:
    if not data[:5].startswith(b"%PDF"):
        return ""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Reading PDFs requires 'pypdf' (pip install pypdf).") from exc
    reader = PdfReader(io.BytesIO(data))
    parts, total = [], 0
    for page in reader.pages[:max_pages]:
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n".join(parts)


# ----- Playwright fallback ----------------------------------------------------#
def playwright_fetch(urls: List[str], timeout_ms: int) -> dict:
    """Render JS-heavy pages with one shared headless Chromium. Returns {url: html}."""
    html_map: dict = {}
    if not urls:
        return html_map
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return html_map
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=HEADERS["User-Agent"], viewport={"width": 1280, "height": 900}
            )
            page = context.new_page()
            for url in urls:
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    page.wait_for_timeout(700)  # let late JS settle a bit
                    html_map[url] = page.content()
                except Exception:
                    continue
            browser.close()
    except Exception:
        return html_map
    return html_map


def run_playwright_in_thread(urls: List[str], timeout_ms: int) -> dict:
    """Run the sync Playwright fallback in a clean thread (no event-loop clashes)."""
    box: dict = {}

    def _go():
        box.update(playwright_fetch(urls, timeout_ms))

    thread = threading.Thread(target=_go, daemon=True)
    thread.start()
    thread.join(timeout=max(60.0, (timeout_ms / 1000.0 + 2) * (len(urls) + 1)))
    return box


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class ResearchAgent:
    """Compiled LangGraph + LLM client + embedding model (+ optional reranker)."""

    def __init__(self, client, embedding_model, config: Optional[ScraperConfig] = None, reranker=None):
        self.client = client
        self.embedding_model = embedding_model
        self.config = config or ScraperConfig()
        self._reranker = reranker
        self._reranker_failed = False
        self._reset_index()
        self.graph = self._build_graph()

    # ----- index store (lives on the agent; numpy/faiss don't belong in state) #
    def _reset_index(self) -> None:
        self._chunks: List[dict] = []
        self._chunk_hashes: set = set()
        self._matrix: Optional[np.ndarray] = None
        self._faiss = None
        self._bm25 = None
        self._indexed_doc_keys: set = set()

    # ----- LLM helpers ------------------------------------------------------ #
    def _chat(self, system: str, user: str, json_mode: bool = False, max_tokens: int = 1024) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(4):
            try:
                kwargs = dict(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                # Drop JSON mode after the first failure in case the backend
                # model doesn't support response_format.
                if json_mode and attempt < 2:
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:  # rate limits, 5xx, unsupported params...
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise last_exc  # type: ignore[misc]

    def _encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        name = (self.config.embedding_model_name or "").lower()
        if is_query and "bge" in name and "zh" not in name:
            # BGE v1.5 models retrieve better with this query instruction.
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        emb = self.embedding_model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb = np.asarray(emb, dtype="float32")
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb

    def _get_reranker(self):
        if not self.config.rerank or self._reranker_failed:
            return self._reranker if self._reranker is not None else None
        if self._reranker is not None:
            return self._reranker
        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.config.rerank_model_name, max_length=512)
            return self._reranker
        except Exception:
            self._reranker_failed = True
            return None

    # ----- evidence rendering ------------------------------------------------ #
    def _render_evidence(
        self, retrieved: List[dict], serper_answer: str, char_cap: int = 1200
    ) -> Tuple[str, List[dict]]:
        items = list(retrieved)
        if serper_answer:
            items.append(
                {
                    "title": "Google answer box (Serper)",
                    "url": "https://www.google.com",
                    "domain": "google answer box",
                    "date": "",
                    "text": serper_answer,
                    "kind": "answer_box",
                }
            )
        lines, sources = [], []
        for n, chunk in enumerate(items, 1):
            head = f"[{n}] {chunk.get('title') or chunk.get('url', '')} — {chunk.get('domain', '')} — {chunk.get('date') or 'date n/a'}"
            lines.append(head + "\n" + (chunk.get("text") or "")[:char_cap])
            sources.append(
                {
                    "n": n,
                    "title": chunk.get("title") or chunk.get("domain") or chunk.get("url", ""),
                    "url": chunk.get("url", ""),
                    "domain": chunk.get("domain", ""),
                }
            )
        return "\n\n".join(lines), sources

    # ===================== Graph nodes ======================================= #
    def plan_node(self, state: State) -> dict:
        """Expand the question into diverse search queries; target known gaps."""
        cfg = self.config
        round_no = state.get("search_round", 0) + 1
        question = state["question"]
        tried = state.get("tried_queries") or []
        missing = (state.get("missing") or "").strip()

        system = "You are a search-query planner for a research agent. Respond with JSON only."
        user = f"""QUESTION: {question}
TODAY: {date.today().isoformat()}
ALREADY_TRIED: {json.dumps(tried[-12:])}
KNOWN_GAPS: {missing or "none yet"}

Write up to {cfg.queries_per_round} diverse Google-style search queries that will surface pages
containing the exact facts needed to answer the QUESTION.
Rules:
- Each query under 12 words; vary wording, angle and likely source type
  (official statistics, news, reference/encyclopedia).
- For time-sensitive facts include the year or "latest".
- At most one query may use operators like site:gov.in or filetype:pdf, and only
  when official data is clearly the best source.
- If KNOWN_GAPS is present, target those gaps first.
- Never repeat anything in ALREADY_TRIED.

Return JSON: {{"queries": ["...", "..."], "note": "one short line on the strategy"}}"""
        queries: List[str] = []
        try:
            data = extract_json(self._chat(system, user, json_mode=True, max_tokens=300))
            queries = [q.strip() for q in (data.get("queries") or []) if isinstance(q, str) and q.strip()]
        except Exception:
            queries = []
        if round_no == 1 and question not in queries:
            queries = [question] + queries
        queries = queries[: max(1, cfg.queries_per_round)] or [question]
        return {
            "search_round": round_no,
            "queries": queries,
            "query": queries[0],
            "tried_queries": tried + queries,
        }

    def search_node(self, state: State) -> dict:
        """Run every planned query; keep snippets; pick a diverse scrape list."""
        cfg = self.config
        seen = {canonical_url(u) for u in (state.get("urls") or [])}
        snippets = list(state.get("snippets") or [])
        serper_answer = state.get("serper_answer", "")
        provider = ""
        candidates: List[dict] = []

        for q in state.get("queries") or [state["question"]]:
            results: List[dict] = []
            if cfg.serper_api_key:
                try:
                    data = serper_search(q, cfg.serper_api_key, cfg.max_results)
                    provider = "Serper (Google)"
                    answer = build_serper_answer(data)
                    if answer:
                        serper_answer = answer
                    for rank, item in enumerate(data.get("organic") or []):
                        if item.get("link"):
                            results.append(
                                {
                                    "url": item["link"],
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                    "date": item.get("date", ""),
                                    "rank": rank,
                                }
                            )
                    for paa in (data.get("peopleAlsoAsk") or [])[:4]:
                        if paa.get("snippet") and paa.get("link"):
                            snippets.append(
                                {
                                    "url": canonical_url(paa["link"]),
                                    "title": paa.get("question", ""),
                                    "text": paa["snippet"],
                                    "date": "",
                                }
                            )
                except Exception:
                    results = []
            if not results:
                try:
                    for rank, item in enumerate(ddg_search(q, cfg.max_results)):
                        results.append({**item, "rank": rank})
                    provider = provider or "DuckDuckGo"
                except Exception:
                    pass

            for r in results:
                url = canonical_url(r["url"])
                dom = domain_of(url)
                if not url or is_blocked(dom):
                    continue
                r["url"], r["domain"] = url, dom
                candidates.append(r)
                if cfg.use_search_snippets and r.get("snippet"):
                    snippets.append(
                        {"url": url, "title": r.get("title", ""), "text": r["snippet"], "date": r.get("date", "")}
                    )

        # Score candidates: search rank + authority boost; dedupe; spread domains.
        best: dict = {}
        for r in candidates:
            if r["url"] in seen:
                continue
            score = 10 - min(int(r.get("rank", 9)), 9)
            if is_authority(r["domain"]):
                score += 4
            if r["url"].lower().split("?")[0].endswith(".pdf"):
                score += 1
            if r["url"] not in best or score > best[r["url"]]["score"]:
                best[r["url"]] = {**r, "score": score}

        ranked = sorted(best.values(), key=lambda x: -x["score"])
        new_urls: List[str] = []
        per_domain: dict = {}
        for r in ranked:
            if len(new_urls) >= cfg.max_pages_per_round:
                break
            if per_domain.get(r["domain"], 0) >= 2:  # at most 2 pages per domain per round
                continue
            per_domain[r["domain"]] = per_domain.get(r["domain"], 0) + 1
            new_urls.append(r["url"])

        # Dedupe snippets, keep the freshest tail.
        unique_snippets, snip_keys = [], set()
        for s in snippets:
            key = (s.get("url", ""), (s.get("text") or "")[:80])
            if key in snip_keys or not s.get("text"):
                continue
            snip_keys.add(key)
            unique_snippets.append(s)

        return {
            "new_urls": new_urls,
            "urls": (state.get("urls") or []) + new_urls,
            "search_provider": provider or "DuckDuckGo",
            "serper_answer": serper_answer,
            "snippets": unique_snippets[-60:],
        }

    # ----- scraping ----------------------------------------------------------- #
    def _fetch_one(self, client: httpx.Client, url: str) -> Optional[dict]:
        response = client.get(url)
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").lower()
        if "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
            text = pdf_to_text(response.content, self.config.pdf_max_pages, self.config.pdf_max_chars)
            return {
                "key": url, "url": url, "title": url.rsplit("/", 1)[-1] or url,
                "date": "", "text": text, "kind": "pdf",
            }
        text, title, dt = extract_main_text(response.text, url)
        return {"key": url, "url": url, "title": title or url, "date": dt, "text": text, "kind": "page"}

    def scrape_node(self, state: State) -> dict:
        cfg = self.config
        timeout_s = max(3.0, cfg.page_timeout_ms / 1000.0)
        documents = list(state.get("documents") or [])
        have_keys = {d.get("key") for d in documents}
        targets = [u for u in (state.get("new_urls") or []) if u not in have_keys]

        fetched: dict = {}
        needs_js: List[str] = []
        if targets:
            with httpx.Client(follow_redirects=True, headers=HEADERS, timeout=timeout_s) as client:
                with ThreadPoolExecutor(max_workers=min(6, len(targets))) as pool:
                    futures = {pool.submit(self._fetch_one, client, u): u for u in targets}
                    for fut in as_completed(futures):
                        url = futures[fut]
                        try:
                            doc = fut.result()
                        except Exception:
                            doc = None
                        if doc and len(doc.get("text") or "") >= cfg.min_extract_chars:
                            fetched[url] = doc
                        elif not url.lower().split("?")[0].endswith(".pdf"):
                            needs_js.append(url)

        if needs_js and cfg.playwright_fallback:
            html_map = run_playwright_in_thread(needs_js, cfg.page_timeout_ms)
            for url, html in html_map.items():
                text, title, dt = extract_main_text(html, url)
                if text and len(text) >= cfg.min_extract_chars:
                    fetched[url] = {
                        "key": url, "url": url, "title": title or url,
                        "date": dt, "text": text, "kind": "page",
                    }

        # Add fetched pages (dedupe near-identical content across mirrors).
        content_hashes = {
            hashlib.md5((d.get("text") or "")[:2000].encode("utf-8", "ignore")).hexdigest()
            for d in documents
        }
        for url in targets:
            doc = fetched.get(url)
            if not doc or not doc.get("text"):
                continue
            doc["text"] = doc["text"][: cfg.per_doc_char_cap]
            digest = hashlib.md5(doc["text"][:2000].encode("utf-8", "ignore")).hexdigest()
            if digest in content_hashes:
                continue
            content_hashes.add(digest)
            doc["domain"] = domain_of(url)
            documents.append(doc)
            have_keys.add(url)

        # Search snippets become tiny cited documents — they often contain the
        # exact fact and survive even when the page itself refuses to load.
        if cfg.use_search_snippets:
            for s in state.get("snippets") or []:
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                key = "snippet::" + s.get("url", "") + "::" + hashlib.md5(text[:80].encode("utf-8", "ignore")).hexdigest()[:10]
                if key in have_keys:
                    continue
                have_keys.add(key)
                documents.append(
                    {
                        "key": key,
                        "url": s.get("url", ""),
                        "title": s.get("title") or s.get("url", ""),
                        "date": s.get("date", ""),
                        "text": text,
                        "kind": "snippet",
                        "domain": domain_of(s.get("url", "")),
                    }
                )

        return {"documents": documents, "n_documents": len(documents)}

    # ----- indexing ------------------------------------------------------------ #
    def index_node(self, state: State) -> dict:
        cfg = self.config
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        new_chunks: List[dict] = []
        for doc in state.get("documents") or []:
            key = doc.get("key") or doc.get("url")
            if key in self._indexed_doc_keys:
                continue
            self._indexed_doc_keys.add(key)
            pieces = [doc["text"]] if doc.get("kind") == "snippet" else splitter.split_text(doc.get("text") or "")
            for piece in pieces:
                piece = piece.strip()
                if doc.get("kind") != "snippet" and len(piece) < 60:
                    continue
                digest = hashlib.md5(re.sub(r"\s+", " ", piece.lower()).encode("utf-8", "ignore")).hexdigest()
                if digest in self._chunk_hashes:
                    continue
                self._chunk_hashes.add(digest)
                new_chunks.append(
                    {
                        "text": piece,
                        "url": doc.get("url", ""),
                        "title": doc.get("title", ""),
                        "date": doc.get("date", ""),
                        "domain": doc.get("domain", ""),
                        "kind": doc.get("kind", "page"),
                    }
                )

        room = cfg.max_total_chunks - len(self._chunks)
        if new_chunks and room > 0:
            new_chunks = new_chunks[:room]
            embeddings = self._encode([c["text"] for c in new_chunks], is_query=False)
            self._matrix = embeddings if self._matrix is None else np.vstack([self._matrix, embeddings])
            self._chunks.extend(new_chunks)
            if faiss is not None:
                self._faiss = faiss.IndexFlatIP(self._matrix.shape[1])  # cosine (normalized)
                self._faiss.add(self._matrix)
            if BM25Okapi is not None and self._chunks:
                self._bm25 = BM25Okapi([tokenize(c["text"]) for c in self._chunks])

        return {"n_chunks": len(self._chunks)}

    # ----- retrieval ------------------------------------------------------------ #
    def retrieve_node(self, state: State) -> dict:
        cfg = self.config
        if not self._chunks or self._matrix is None:
            return {"retrieved": [], "retrieved_chunks": []}

        question = state["question"]
        queries = [question] + [q for q in (state.get("queries") or []) if q and q != question]
        n_candidates = min(len(self._chunks), max(cfg.retrieve_k, cfg.top_k * 3))

        fused: dict = {}
        for q in queries[:4]:
            # Dense (cosine over normalized embeddings)
            query_vec = self._encode([q], is_query=True)
            if self._faiss is not None:
                _, idx = self._faiss.search(query_vec, n_candidates)
                dense_ranked = [int(i) for i in idx[0] if i >= 0]
            else:
                sims = self._matrix @ query_vec[0]
                dense_ranked = [int(i) for i in np.argsort(-sims)[:n_candidates]]
            for rank, i in enumerate(dense_ranked):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            # Lexical (BM25) — catches exact numbers, names, codes that
            # small embedding models blur together.
            if self._bm25 is not None:
                scores = self._bm25.get_scores(tokenize(q))
                for rank, i in enumerate(np.argsort(-scores)[:n_candidates]):
                    i = int(i)
                    if scores[i] <= 0:
                        break
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)

        candidate_idx = sorted(fused, key=lambda i: -fused[i])[: cfg.retrieve_k]
        candidates = [dict(self._chunks[i], score=float(fused[i])) for i in candidate_idx]

        # Cross-encoder rerank against the ORIGINAL question.
        reranker = self._get_reranker()
        if reranker is not None and candidates:
            try:
                pairs = [(question, c["text"][:1500]) for c in candidates]
                ce_scores = np.asarray(reranker.predict(pairs), dtype="float32")
                order = list(np.argsort(-ce_scores))
                candidates = [dict(candidates[i], score=float(ce_scores[i])) for i in order]
            except Exception:
                pass  # keep RRF order

        # Final evidence set with per-domain diversity (avoid one site dominating).
        picked: List[dict] = []
        per_domain: dict = {}
        for c in candidates:
            dom = c.get("domain", "")
            if per_domain.get(dom, 0) >= cfg.max_chunks_per_domain:
                continue
            per_domain[dom] = per_domain.get(dom, 0) + 1
            picked.append(c)
            if len(picked) >= cfg.top_k:
                break
        if len(picked) < min(cfg.top_k, len(candidates)):  # backfill if starved
            for c in candidates:
                if c not in picked:
                    picked.append(c)
                if len(picked) >= cfg.top_k:
                    break

        rendered = [
            f"[{c.get('domain') or 'source'}] {c.get('title', '')} ({c.get('date') or 'date n/a'})\n{c['text']}"
            for c in picked
        ]
        return {"retrieved": picked, "retrieved_chunks": rendered}

    # ----- sufficiency audit ------------------------------------------------------ #
    def evaluate_node(self, state: State) -> dict:
        cfg = self.config
        round_no = state.get("search_round", 1)
        retrieved = state.get("retrieved") or []

        if not retrieved:
            if round_no >= cfg.max_searches:
                return {"enough_info": True, "missing": "no usable evidence was found"}
            return {"enough_info": False, "missing": "no usable evidence retrieved; try different sources or phrasing"}

        evidence, _ = self._render_evidence(retrieved, state.get("serper_answer", ""), char_cap=700)
        system = "You are a strict research auditor. Respond with JSON only."
        user = f"""QUESTION: {state['question']}
TODAY: {date.today().isoformat()}

EVIDENCE:
{evidence}

Can the QUESTION be answered precisely from the EVIDENCE alone — correct entities,
correct numbers with units, and the correct time period? Be strict: vague or stale
evidence is NOT enough.

Return JSON: {{"enough_info": true/false,
"missing": "what exactly is still missing (empty if nothing)",
"followup_query": "one search query that would fill the gap (empty if none)"}}"""
        enough, missing = False, ""
        try:
            data = extract_json(self._chat(system, user, json_mode=True, max_tokens=300))
            enough = bool(data.get("enough_info"))
            missing = str(data.get("missing") or data.get("followup_query") or "")
        except Exception:
            enough = True  # if the auditor itself fails, fall through to answering
        if round_no >= cfg.max_searches:
            enough = True
        return {"enough_info": enough, "missing": missing}

    # ----- answering ------------------------------------------------------------- #
    def answer_node(self, state: State) -> dict:
        retrieved = state.get("retrieved") or []
        evidence, sources = self._render_evidence(retrieved, state.get("serper_answer", ""))
        system = (
            "You are a meticulous research analyst. You answer using ONLY the numbered "
            "EVIDENCE provided. You never use outside knowledge and you never guess."
        )
        user = f"""TODAY: {date.today().isoformat()}
QUESTION: {state['question']}

EVIDENCE:
{evidence if evidence else "(no evidence was retrieved)"}

Write the answer following ALL of these rules:
1. Start with a direct 1–2 sentence answer, then brief supporting detail. No filler.
2. Every sentence stating a fact, number, name or date MUST end with citations
   like [2] or [1,4] pointing to EVIDENCE items.
3. Copy numbers exactly as written in the evidence, with units and the time
   period (e.g. "as of 2024", "in FY 2023-24"). Never round, average or convert.
4. Prefer facts confirmed by two or more independent sources. If a key fact has
   only one source, attribute it ("according to ...").
5. If sources disagree, report each value with its citation and prefer the most
   recent / most authoritative — never invent a compromise figure.
6. The Google answer box (if present) is a hint, not proof — state its claim as
   fact only if another source corroborates it; otherwise attribute it.
7. If the evidence does not fully answer the question, answer what IS supported
   and state explicitly what could not be verified. Never fill gaps from memory."""
        answer = self._chat(system, user, max_tokens=900).strip()
        return {"answer": answer, "sources": sources}

    # ----- verification ------------------------------------------------------------ #
    def verify_node(self, state: State) -> dict:
        cfg = self.config
        retrieved = state.get("retrieved") or []
        evidence, _ = self._render_evidence(retrieved, state.get("serper_answer", ""))
        draft = state.get("answer") or ""

        system = "You are a strict, adversarial fact-checker. Respond with JSON only."
        user = f"""QUESTION: {state['question']}

EVIDENCE:
{evidence}

DRAFT ANSWER:
{draft}

Check every factual claim in the DRAFT against the EVIDENCE only.
A claim is supported only if the evidence states it with the same numbers,
entities, units and time period.

Return JSON:
{{"verdict": "pass" or "fail",
"unsupported_claims": ["verbatim claims not backed by the evidence"],
"single_source_claims": ["key claims backed by only one domain"],
"confidence": "high" or "medium" or "low",
"reason": "one line"}}

confidence: high = key facts corroborated by 2+ independent domains;
medium = supported but mostly single-source; low = gaps or contradictions."""
        verdict, unsupported, single_source, confidence, reason = "pass", [], [], "medium", ""
        try:
            data = extract_json(self._chat(system, user, json_mode=True, max_tokens=500))
            verdict = str(data.get("verdict", "pass")).lower()
            unsupported = [str(x) for x in (data.get("unsupported_claims") or [])]
            single_source = [str(x) for x in (data.get("single_source_claims") or [])]
            confidence = str(data.get("confidence") or ("high" if verdict == "pass" else "low")).lower()
            reason = str(data.get("reason") or "")
        except Exception:
            pass
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        verification = {
            "verdict": verdict if verdict in ("pass", "fail") else ("fail" if unsupported else "pass"),
            "unsupported_claims": unsupported,
            "single_source_claims": single_source,
            "reason": reason,
        }
        result: dict = {"verification": verification, "confidence": confidence}
        round_no = state.get("search_round", 1)

        if verification["verdict"] == "fail" and unsupported and round_no < cfg.max_searches:
            # There's budget left: go gather evidence for the failed claims.
            result["enough_info"] = False
            result["missing"] = "find sources confirming or refuting: " + "; ".join(unsupported[:3])
            return result

        if verification["verdict"] == "fail" and unsupported:
            # Out of budget: strip/qualify unsupported claims instead of shipping them.
            fix_user = (
                f"EVIDENCE:\n{evidence}\n\nDRAFT:\n{draft}\n\nUNSUPPORTED CLAIMS:\n- "
                + "\n- ".join(unsupported)
                + "\n\nRewrite the draft so it contains ONLY claims supported by the evidence, "
                "keeping the [n] citations. Explicitly note what could not be verified. Be concise."
            )
            try:
                fixed = self._chat(
                    "You repair research answers so every claim is evidence-backed.",
                    fix_user,
                    max_tokens=900,
                ).strip()
                if fixed:
                    result["answer"] = fixed
            except Exception:
                pass
            result["confidence"] = "low" if confidence == "low" else "medium"

        result["enough_info"] = True
        return result

    # ----- finalize ------------------------------------------------------------- #
    def finalize_node(self, state: State) -> dict:
        answer = (state.get("answer") or "_No answer was produced._").strip()
        if self.config.verify_answer:
            confidence = (state.get("confidence") or "medium").lower()
        else:
            confidence = state.get("confidence") or "unverified"
        badge = {
            "high": "🟢 High — key facts corroborated by multiple independent sources",
            "medium": "🟡 Medium — supported by evidence, but mostly single-source",
            "low": "🔴 Low — gaps or contradictions remain; treat with care",
            "unverified": "⚪ Not verified (verification pass disabled)",
        }.get(confidence, confidence)

        # List the sources the answer actually cites (fall back to all).
        sources = state.get("sources") or []
        cited_nums = set()
        for group in re.findall(r"\[([\d,\s]+)\]", answer):
            for token in group.split(","):
                token = token.strip()
                if token.isdigit():
                    cited_nums.add(int(token))
        listed = [s for s in sources if s.get("n") in cited_nums] or sources

        lines = [answer, "", f"**Confidence:** {badge}"]
        verification = state.get("verification") or {}
        if verification.get("single_source_claims"):
            lines.append(
                "_Single-source claims (worth double-checking): "
                + "; ".join(verification["single_source_claims"][:3])
                + "_"
            )
        if listed:
            lines += ["", "**Sources**"]
            grouped: List[Tuple[str, List[dict]]] = []  # (url-or-blank, [sources])
            by_url: dict = {}
            for s in listed:
                u = s.get("url") or ""
                if u and u in by_url:
                    by_url[u].append(s)
                    continue
                entry = [s]
                if u:
                    by_url[u] = entry
                grouped.append((u, entry))
            for u, group in grouped:
                nums = ", ".join(str(s["n"]) for s in group)
                label = next((s.get("title") for s in group if s.get("title")), None) or (
                    group[0].get("domain") or u or "source"
                )
                if u:
                    lines.append(f"{nums}. [{label}]({u})")
                else:
                    lines.append(f"{nums}. {label}")
        return {"answer": "\n".join(lines), "confidence": confidence}

    # ===================== Graph wiring ======================================== #
    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("plan", self.plan_node)
        graph.add_node("search", self.search_node)
        graph.add_node("scrape", self.scrape_node)
        graph.add_node("index", self.index_node)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("evaluate", self.evaluate_node)
        graph.add_node("answer", self.answer_node)
        graph.add_node("finalize", self.finalize_node)

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "scrape")
        graph.add_edge("scrape", "index")
        graph.add_edge("index", "retrieve")
        graph.add_edge("retrieve", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            lambda s: "answer" if s.get("enough_info") else "plan",
            {"answer": "answer", "plan": "plan"},
        )
        if self.config.verify_answer:
            graph.add_node("verify", self.verify_node)
            graph.add_edge("answer", "verify")
            graph.add_conditional_edges(
                "verify",
                lambda s: "finalize" if s.get("enough_info", True) else "plan",
                {"finalize": "finalize", "plan": "plan"},
            )
        else:
            graph.add_edge("answer", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    # ===================== Public runners ====================================== #
    def stream_research(self, question: str) -> Iterator[Tuple[str, dict]]:
        """Run the graph, yielding (kind, payload) events.

        kind == "update": payload is {node_name: state_delta_from_that_node}
        kind == "error":  payload is the raised Exception
        """
        self._reset_index()
        initial: State = {
            "question": question.strip(),
            "query": question.strip(),
            "search_round": 0,
            "urls": [],
            "documents": [],
            "snippets": [],
            "tried_queries": [],
            "queries": [],
            "missing": "",
        }
        try:
            for update in self.graph.stream(initial, config={"recursion_limit": 200}):
                yield ("update", update)
        except Exception as exc:  # surfaced to the UI
            yield ("error", exc)

    def research(self, question: str) -> dict:
        """Run the pipeline to completion and return the merged final state."""
        merged: dict = {}
        for kind, payload in self.stream_research(question):
            if kind == "error":
                raise payload
            merged.update(next(iter(payload.values())))
        return merged
