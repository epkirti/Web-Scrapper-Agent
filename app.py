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
from sentence_transformers import SentenceTransformer, CrossEncoder

from scraper import ResearchAgent, ScraperConfig, STEP_LABELS
from google_ai_overview import fetch_ai_overview_sync

# Persistent Chrome profile for the Quick-answer fetcher. Reusing it across runs
# builds up cookies/trust. Kept inside the project (the C: drive is full here).
CHROME_PROFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chrome-profile")

RAG_MODE = "Deep research (RAG)"
OVERVIEW_MODE = "Quick answer"

st.set_page_config(page_title="Agentic Web Research", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------- #
# Cached heavy resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


@st.cache_resource(show_spinner="Loading reranker (first run downloads ~80MB)…")
def get_reranker(name: str) -> CrossEncoder:
    return CrossEncoder(name, max_length=512)


# --------------------------------------------------------------------------- #
# Sidebar — configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Mode",
        [RAG_MODE, OVERVIEW_MODE],
        help=(
            "Deep research: searches multiple pages and answers with RAG.\n\n"
            "Quick answer: returns a short, direct answer."
        ),
    )

    api_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get one at https://console.groq.com/keys",
    )
    model = st.text_input("Groq model", value="llama-3.3-70b-versatile")

    if mode == RAG_MODE:
        embedding_model_name = st.text_input("Embedding model", value="all-MiniLM-L6-v2")

        serper_api_key = st.text_input(
            "Serper API key (Google search)",
            type="password",
            value=os.getenv("SERPER_API_KEY", ""),
            help="Free key at https://serper.dev — gives Google results. Leave blank to use DuckDuckGo.",
        )

        reranker_model_name = st.text_input("Reranker (cross-encoder)", value="cross-encoder/ms-marco-MiniLM-L-6-v2")

        max_searches = st.slider("Max search rounds", 1, 10, 3)
        max_results = st.slider("Results per search", 3, 20, 10)

        with st.expander("Advanced"):
            chunk_size = st.number_input("Chunk size", 100, 2000, 700, step=50)
            chunk_overlap = st.number_input("Chunk overlap", 0, 500, 120, step=20)
            top_k = st.number_input("Chunks kept after rerank (top-k)", 1, 20, 5)
            first_stage_k = st.number_input("Candidates before rerank", 5, 60, 20, step=5)
            min_similarity = st.slider("Min cosine similarity (abstain floor)", 0.0, 0.6, 0.30, step=0.05)
            rerank_min_score = st.slider("Min rerank score (logit floor)", -10.0, 10.0, 0.0, step=0.5)
            page_timeout_ms = st.number_input("Page load timeout (ms)", 3000, 60000, 15000, step=1000)
            pdf_max_pages = st.number_input("Max PDF pages to read", 5, 500, 50, step=5)

        st.divider()
        st.caption("search → scrape → chunk → embed → cosine retrieve → rerank → cited answer → verify")
    else:
        with st.expander("Advanced"):
            overview_timeout_ms = st.number_input(
                "Answer wait timeout (ms)", 10000, 60000, 25000, step=1000,
            )


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


# --------------------------------------------------------------------------- #
# Quick answer mode
#
# Internally: silently fetch the source text, then have Groq distill it into a
# short, direct answer. The UI never reveals where the text came from.
# --------------------------------------------------------------------------- #
if run and mode == OVERVIEW_MODE:
    if not api_key.strip():
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    answer = ""
    with st.spinner("Finding the answer…"):
        try:
            result = fetch_ai_overview_sync(
                query.strip(),
                user_data_dir=CHROME_PROFILE,
                hidden=True,           # real browser, parked off-screen
                headless=False,
                timeout_ms=int(overview_timeout_ms),
            )
        except Exception:  # noqa: BLE001 - never surface internals to the user
            result = {"text": "", "unavailable": True}

        raw = (result.get("text") or "").strip()
        if raw:
            prompt = (
                "Answer the user's question in a few short, clear sentences. "
                "Be direct and to the point. Use only the information below, but "
                "do NOT mention the information, its source, or that it was "
                "provided to you — just answer naturally.\n\n"
                f"Question: {query.strip()}\n\n"
                f"Information:\n{raw}\n\nAnswer:"
            )
            try:
                groq_client = Groq(api_key=api_key.strip())
                resp = groq_client.chat.completions.create(
                    model=model.strip(),
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "invalid_api_key" in msg or "401" in msg:
                    st.error("Your Groq API key was rejected. Check it in the sidebar.")
                else:
                    st.error("Couldn't generate the answer. Please try again.")
                st.stop()

    st.subheader("Answer")
    if answer:
        st.markdown(answer)
    else:
        st.info("Sorry, I couldn't find an answer for that. Please try again in a moment.")
    st.stop()


if run:
    if not api_key.strip():
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    config = ScraperConfig(
        model=model.strip(),
        embedding_model_name=embedding_model_name.strip(),
        reranker_model_name=reranker_model_name.strip(),
        max_searches=int(max_searches),
        max_results=int(max_results),
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        top_k=int(top_k),
        first_stage_k=int(first_stage_k),
        min_similarity=float(min_similarity),
        rerank_min_score=float(rerank_min_score),
        page_timeout_ms=int(page_timeout_ms),
        pdf_max_pages=int(pdf_max_pages),
        serper_api_key=serper_api_key.strip(),
    )

    embedding_model = get_embedding_model(config.embedding_model_name)
    reranker = get_reranker(config.reranker_model_name) if config.reranker_model_name else None
    agent = ResearchAgent(Groq(api_key=api_key.strip()), embedding_model, config, reranker=reranker)

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
            elif node == "retrieve":
                rc = node_state.get("retrieved_chunks") or []
                if rc:
                    mean_sim = sum(c.get("sim", 0.0) for c in rc) / len(rc)
                    st.write(f"{label} · kept {len(rc)} (mean cosine {mean_sim:.2f})")
                else:
                    st.write(f"{label} · nothing cleared the relevance floor")
            elif node == "evaluate":
                verdict = "sufficient ✅" if node_state.get("enough_info") else "needs more 🔁"
                st.write(f"{label} → {verdict}")
            else:
                st.write(label)

        status.update(label="Done", state="complete")

    # ----- Results -------------------------------------------------------- #
    st.subheader("Answer")
    st.markdown(merged.get("answer") or "_No answer was produced._")

    # --- Trust panel: make honesty visible -------------------------------- #
    score = merged.get("confidence_score")
    corr = merged.get("corroboration_max", 0)
    if merged.get("abstained"):
        st.error("🚫 Abstained — the retrieved sources did not contain enough to answer.")
    elif score is not None:
        if score >= 0.75:
            st.success(f"🟢 High confidence · score {score:.2f}")
        elif score >= 0.4:
            st.warning(f"🟡 Partial confidence · score {score:.2f}")
        else:
            st.error(f"🔴 Low / unverified · score {score:.2f}")
        st.caption(
            f"Best-corroborated claim confirmed by **{corr}** independent domain(s). "
            "Confidence is a calibrated heuristic — every claim is auditable below, never a guarantee."
        )

    claims = merged.get("claims") or []
    if claims:
        with st.expander(f"🔍 Claim-by-claim verification ({len(claims)})", expanded=True):
            rows = []
            for c in claims:
                srcs = ", ".join(f"[{n}]" for n in c.get("sources", [])) or "—"
                rows.append({
                    "Claim": c.get("text", ""),
                    "Sources": srcs,
                    "Status": c.get("status", "—"),
                })
            st.table(rows)

    if merged.get("unverified_claims") or merged.get("conflicts"):
        with st.expander("⚠️ Could NOT be verified", expanded=True):
            for u in merged.get("unverified_claims") or []:
                st.markdown(f"- {u}")
            for cf in merged.get("conflicts") or []:
                st.markdown(f"- ⚔️ **Conflict:** {cf}")

    if merged.get("serper_answer"):
        with st.expander("📌 Google instant answer (fed into the context)", expanded=False):
            st.markdown(merged["serper_answer"])

    chunks = merged.get("retrieved_chunks") or []
    # Distinct real sources actually used (excludes the Serper aggregator doc).
    src_urls = []
    for c in chunks:
        u = c.get("url")
        if u and u not in src_urls:
            src_urls.append(u)

    col1, col2 = st.columns(2)
    with col1:
        with st.expander(f"🔗 Sources used ({len(src_urls)})", expanded=False):
            if src_urls:
                for u in src_urls:
                    st.markdown(f"- [{u}]({u})" if u.startswith("http") else f"- {u}")
            else:
                st.write("No sources cleared the relevance floor.")
    with col2:
        with st.expander(f"📥 Retrieved context ({len(chunks)} chunks)", expanded=False):
            if chunks:
                st.caption("Scores are _relevance_ measures, not probabilities.")
                for i, chunk in enumerate(chunks, 1):
                    u = chunk.get("url", "")
                    link = f"[{chunk.get('domain', u)}]({u})" if u.startswith("http") else chunk.get("domain", u)
                    scores = f"cosine {chunk.get('sim', 0):.2f}"
                    if "rerank" in chunk:
                        scores += f" · rerank {chunk['rerank']:.2f}"
                    st.markdown(f"**[{i}]** {link} · _{scores}_")
                    st.write(chunk.get("text", ""))
                    st.divider()
            else:
                st.write("No context was retrieved.")

    if refined_queries:
        with st.expander("♻️ Refined queries"):
            st.write(f"Original: `{query.strip()}`")
            for i, q in enumerate(refined_queries, 1):
                st.write(f"Round {i}: `{q}`")
