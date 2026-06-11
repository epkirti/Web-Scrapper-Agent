"""Streamlit front-end for the agentic web-research RAG pipeline.

Run with:
    streamlit run app.py
"""

import os

# faiss and torch each bundle their own OpenMP runtime; on macOS loading both
# can segfault. These must be set BEFORE torch/faiss are imported (below).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st
from groq import Groq
from sentence_transformers import CrossEncoder, SentenceTransformer

from scraper import ResearchAgent, ScraperConfig, STEP_LABELS

st.set_page_config(page_title="Agentic Web Research", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------- #
# Cached heavy resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


@st.cache_resource(show_spinner="Loading reranker model…")
def get_reranker(name: str) -> CrossEncoder:
    return CrossEncoder(name, max_length=512)


# --------------------------------------------------------------------------- #
# Sidebar — configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get one at https://console.groq.com/keys",
    )
    model = st.text_input("Groq model", value="llama-3.3-70b-versatile")
    embedding_model_name = st.text_input(
        "Embedding model",
        value="all-MiniLM-L6-v2",
        help="For Hindi/Hinglish questions try `paraphrase-multilingual-MiniLM-L12-v2`.",
    )

    serper_api_key = st.text_input(
        "Serper API key (Google search)",
        type="password",
        value=os.getenv("SERPER_API_KEY", ""),
        help="Free key at https://serper.dev — gives Google results. Leave blank to use DuckDuckGo.",
    )

    max_searches = st.slider("Max research rounds", 1, 10, 4)
    queries_per_round = st.slider(
        "Search queries per round", 1, 5, 3,
        help="The planner rewrites your question into this many diverse queries each round.",
    )
    max_results = st.slider("Results per search", 3, 20, 10)

    st.subheader("🎯 Accuracy")
    rerank = st.checkbox(
        "Cross-encoder reranking", value=True,
        help="Re-scores retrieved chunks against your exact question. Slower, much more precise.",
    )
    rerank_model_name = st.text_input(
        "Reranker model", value="cross-encoder/ms-marco-MiniLM-L-6-v2", disabled=not rerank,
    )
    verify_answer = st.checkbox(
        "Verify answer (fact-check pass)", value=True,
        help="An adversarial checker audits every claim; failures trigger more research or a rewrite.",
    )

    with st.expander("Advanced"):
        chunk_size = st.number_input("Chunk size", 100, 2000, 800, step=50)
        chunk_overlap = st.number_input("Chunk overlap", 0, 500, 150, step=25)
        top_k = st.number_input("Evidence chunks (top-k)", 1, 20, 6)
        retrieve_k = st.number_input(
            "Rerank candidates (retrieve-k)", 5, 100, 24, step=1,
            help="Hybrid (FAISS + BM25) candidates fed to the reranker.",
        )
        max_pages_per_round = st.number_input("Pages scraped per round", 2, 25, 8)
        page_timeout_ms = st.number_input("Page load timeout (ms)", 3000, 60000, 15000, step=1000)
        pdf_max_pages = st.number_input("Max PDF pages to read", 5, 500, 50, step=5)

    st.divider()
    st.caption(
        "Pipeline: plan → search → scrape → hybrid index (FAISS + BM25) → "
        "rerank → audit → cited answer → fact-check → finalize"
    )


# --------------------------------------------------------------------------- #
# Main — query & run
# --------------------------------------------------------------------------- #
st.title("🔎 Agentic Web Research Assistant")
st.caption(
    "LangGraph agent that plans searches, scrapes the web, retrieves with hybrid "
    "search + reranking, and writes a **cited, fact-checked** answer."
)

query = st.text_input(
    "Your question",
    placeholder="e.g. How much wheat is produced in Madhya Pradesh?",
)
run = st.button("Research", type="primary", disabled=not query.strip())

if run:
    if not api_key.strip():
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    config = ScraperConfig(
        model=model.strip(),
        embedding_model_name=embedding_model_name.strip(),
        max_searches=int(max_searches),
        max_results=int(max_results),
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        top_k=int(top_k),
        page_timeout_ms=int(page_timeout_ms),
        pdf_max_pages=int(pdf_max_pages),
        serper_api_key=serper_api_key.strip(),
        queries_per_round=int(queries_per_round),
        retrieve_k=int(retrieve_k),
        max_pages_per_round=int(max_pages_per_round),
        rerank=bool(rerank),
        rerank_model_name=rerank_model_name.strip(),
        verify_answer=bool(verify_answer),
    )

    embedding_model = get_embedding_model(config.embedding_model_name)
    reranker = get_reranker(config.rerank_model_name) if config.rerank else None
    agent = ResearchAgent(Groq(api_key=api_key.strip()), embedding_model, config, reranker=reranker)

    merged: dict = {}
    rounds: list = []  # [(round_no, [queries])]

    with st.status("Researching…", expanded=True) as status:
        for kind, payload in agent.stream_research(query.strip()):
            if kind == "error":
                status.update(label="Research failed", state="error")
                st.exception(payload)
                st.stop()

            node, node_state = next(iter(payload.items()))
            merged.update(node_state)
            label = STEP_LABELS.get(node, node)

            if node == "plan":
                round_no = node_state.get("search_round", len(rounds) + 1)
                queries = node_state.get("queries") or []
                rounds.append((round_no, queries))
                st.write(f"**Round {round_no}/{config.max_searches}** — {label}")
                for q in queries:
                    st.write(f"&nbsp;&nbsp;&nbsp;• `{q}`")
            elif node == "search":
                n_new = len(node_state.get("new_urls") or [])
                provider = node_state.get("search_provider", "")
                st.write(f"{label} via **{provider}** · {n_new} new pages queued")
            elif node == "scrape":
                st.write(f"{label} · {node_state.get('n_documents', 0)} documents total")
            elif node == "index":
                st.write(f"{label} · {node_state.get('n_chunks', 0)} chunks indexed")
            elif node == "retrieve":
                st.write(f"{label} · {len(node_state.get('retrieved') or [])} evidence chunks selected")
            elif node == "evaluate":
                if node_state.get("enough_info"):
                    st.write(f"{label} → sufficient ✅")
                else:
                    missing = (node_state.get("missing") or "").strip()
                    st.write(f"{label} → needs more 🔁" + (f" — _{missing}_" if missing else ""))
            elif node == "verify":
                verdict = (node_state.get("verification") or {}).get("verdict", "pass")
                conf = node_state.get("confidence", "")
                if node_state.get("enough_info", True):
                    icon = "✅" if verdict == "pass" else "🛠️ rewrote unsupported claims"
                    st.write(f"{label} → {verdict} {icon} · confidence: **{conf}**")
                else:
                    st.write(f"{label} → fail ❌ · gathering more evidence for the flagged claims 🔁")
            else:
                st.write(label)

        status.update(label="Done", state="complete")

    # ----- Results -------------------------------------------------------- #
    confidence = (merged.get("confidence") or "").lower()
    if confidence == "high":
        st.success("🟢 High confidence — key facts corroborated by multiple independent sources.")
    elif confidence == "medium":
        st.warning("🟡 Medium confidence — supported by evidence, but mostly single-source.")
    elif confidence == "low":
        st.error("🔴 Low confidence — gaps or contradictions remain. Treat with care.")

    st.subheader("Answer")
    st.markdown(merged.get("answer") or "_No answer was produced._")

    verification = merged.get("verification") or {}
    if verification:
        with st.expander("🔬 Verification report", expanded=bool(verification.get("unsupported_claims"))):
            st.write(f"**Verdict:** {verification.get('verdict', '—')}")
            if verification.get("reason"):
                st.write(f"**Reason:** {verification['reason']}")
            if verification.get("unsupported_claims"):
                st.write("**Claims that failed verification (removed/qualified):**")
                for c in verification["unsupported_claims"]:
                    st.write(f"- {c}")
            if verification.get("single_source_claims"):
                st.write("**Single-source claims (worth double-checking):**")
                for c in verification["single_source_claims"]:
                    st.write(f"- {c}")

    if merged.get("serper_answer"):
        with st.expander("📌 Google instant answer (used as a hint, not as proof)", expanded=False):
            st.markdown(merged["serper_answer"])

    urls = merged.get("urls") or []
    chunks = merged.get("retrieved_chunks") or []

    col1, col2 = st.columns(2)
    with col1:
        with st.expander(f"🔗 Pages scraped ({len(urls)})", expanded=False):
            if urls:
                for u in urls:
                    st.markdown(f"- [{u}]({u})")
            else:
                st.write("No pages were scraped.")
    with col2:
        with st.expander(f"📥 Evidence given to the LLM ({len(chunks)} chunks)", expanded=False):
            if chunks:
                for chunk in chunks:
                    st.markdown(chunk)
                    st.divider()
            else:
                st.write("No context was retrieved.")

    if rounds:
        with st.expander("♻️ Search plan by round"):
            st.write(f"Original question: `{query.strip()}`")
            for round_no, queries in rounds:
                st.write(f"**Round {round_no}:**")
                for q in queries:
                    st.write(f"- `{q}`")
