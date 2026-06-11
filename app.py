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
from sentence_transformers import SentenceTransformer

from scraper import ResearchAgent, ScraperConfig, STEP_LABELS

st.set_page_config(page_title="Agentic Web Research", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------- #
# Cached heavy resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


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
    embedding_model_name = st.text_input("Embedding model", value="all-MiniLM-L6-v2")

    serper_api_key = st.text_input(
        "Serper API key (Google search)",
        type="password",
        value=os.getenv("SERPER_API_KEY", ""),
        help="Free key at https://serper.dev — gives Google results. Leave blank to use DuckDuckGo.",
    )

    max_searches = st.slider("Max search rounds", 1, 10, 5)
    max_results = st.slider("Results per search", 3, 20, 10)

    with st.expander("Advanced"):
        chunk_size = st.number_input("Chunk size", 100, 2000, 500, step=50)
        chunk_overlap = st.number_input("Chunk overlap", 0, 500, 100, step=25)
        top_k = st.number_input("Chunks retrieved (top-k)", 1, 20, 5)
        page_timeout_ms = st.number_input("Page load timeout (ms)", 3000, 60000, 15000, step=1000)
        pdf_max_pages = st.number_input("Max PDF pages to read", 5, 500, 50, step=5)

    st.divider()
    st.caption("Pipeline: search → scrape → parse → chunk → embed → FAISS → retrieve → answer")


# --------------------------------------------------------------------------- #
# Main — query & run
# --------------------------------------------------------------------------- #
st.title("🔎 Agentic Web Research Assistant")
st.caption("LangGraph agent that searches the web, scrapes pages, and answers with RAG over what it finds.")

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
    )

    embedding_model = get_embedding_model(config.embedding_model_name)
    agent = ResearchAgent(Groq(api_key=api_key.strip()), embedding_model, config)

    merged: dict = {}
    refined_queries: list = []
    round_no = 0

    with st.status("Researching…", expanded=True) as status:
        for kind, payload in agent.stream_research(query.strip()):
            if kind == "error":
                status.update(label="Research failed", state="error")
                st.exception(payload)
                st.stop()

            node, node_state = next(iter(payload.items()))
            merged.update(node_state)
            label = STEP_LABELS.get(node, node)

            if node == "search":
                round_no += 1
                n_urls = len(node_state.get("urls") or [])
                provider = node_state.get("search_provider", "")
                st.write(f"**Round {round_no}** — {label} via {provider} · found {n_urls} URLs")
            elif node == "research":
                refined = (node_state.get("query") or "").strip()
                refined_queries.append(refined)
                st.write(f"{label}: `{refined}`")
            elif node == "evaluate":
                verdict = "sufficient ✅" if node_state.get("enough_info") else "needs more 🔁"
                st.write(f"{label} → {verdict}")
            else:
                st.write(label)

        status.update(label="Done", state="complete")

    # ----- Results -------------------------------------------------------- #
    st.subheader("Answer")
    st.markdown(merged.get("answer") or "_No answer was produced._")

    urls = merged.get("urls") or []
    chunks = merged.get("retrieved_chunks") or []

    col1, col2 = st.columns(2)
    with col1:
        with st.expander(f"🔗 Sources ({len(urls)})", expanded=False):
            if urls:
                for u in urls:
                    st.markdown(f"- [{u}]({u})")
            else:
                st.write("No sources were scraped.")
    with col2:
        with st.expander(f"📥 Retrieved context ({len(chunks)} chunks)", expanded=False):
            if chunks:
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(chunk)
                    st.divider()
            else:
                st.write("No context was retrieved.")

    if refined_queries:
        with st.expander("♻️ Refined queries"):
            st.write(f"Original: `{query.strip()}`")
            for i, q in enumerate(refined_queries, 1):
                st.write(f"Round {i}: `{q}`")
