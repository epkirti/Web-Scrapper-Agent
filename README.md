# Agentic Web Research Assistant

A Streamlit app wrapping the LangGraph web-scraping RAG agent from
`WebScrapper_Agents.ipynb`.

Given a question, the agent:

1. **Searches** the web (Google via Serper, or DuckDuckGo fallback) + pulls Google's instant/AI answer
2. **Scrapes** the result pages (Playwright / headless Chromium) and reads **PDFs** (pypdf)
3. **Parses** HTML to text (BeautifulSoup), accumulating evidence across search rounds
4. **Chunks** the text with provenance — each chunk keeps its `{text, url, domain}` (LangChain splitter, sentence-aware)
5. **Embeds** the chunks (SentenceTransformers `all-MiniLM-L6-v2`, L2-normalized)
6. **Retrieves** with cosine similarity (FAISS `IndexFlatIP`) above an abstention floor, then **reranks**
   the top candidates with a cross-encoder (`ms-marco-MiniLM-L-6-v2`) and keeps the best *k*
7. **Evaluates** sufficiency (Groq LLM). If weak, it **refines the query** from the original question and
   loops; otherwise it **answers** strictly from the retrieved context, with mandatory `[n]` **citations**
8. **Verifies** each claim against its cited chunks, counts **cross-domain corroboration**, flags conflicts,
   and produces a **confidence score** — or explicitly **abstains** ("could not verify").

### Accuracy & trust

The pipeline is tuned for *verifiable* answers, not blind confidence. Generation is deterministic
(`temperature=0`); every factual sentence cites the numbered source it came from; a separate verify pass
checks whether each claim is actually supported by its cited chunk and whether ≥2 independent domains agree.
The UI shows a confidence badge, a per-claim verification table, the corroboration count, and any unverified
or conflicting claims — so you can audit every statement against the exact chunk and URL behind it. It says
**"I could not find sufficient information"** rather than guessing. (No system, including Google, can
guarantee 100% correctness — if a fact isn't in any retrieved source, it abstains.)

## Setup

```bash
cd "Web Scrapper"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium          # one-time browser download
```

## Run

```bash
streamlit run app.py
```

Then in the sidebar:

- Paste your **Groq API key** (get one at <https://console.groq.com/keys>), or
  set it once via `export GROQ_API_KEY=...` before launching.
- *(Optional)* Paste a **Serper API key** (free at <https://serper.dev>) to use
  **Google** search instead of DuckDuckGo — much better source quality. Leave it
  blank to fall back to DuckDuckGo (no key needed). Can also be set via
  `export SERPER_API_KEY=...`.
- Optionally tweak the model, number of search rounds, chunking, and retrieval
  settings.

Enter a question (e.g. *“How much wheat is produced in Madhya Pradesh?”*) and
click **Research**. Progress streams live; the answer, sources, retrieved
context, and any refined queries are shown when it finishes.

## Files

| File             | Purpose                                                        |
| ---------------- | -------------------------------------------------------------- |
| `app.py`         | Streamlit UI                                                   |
| `scraper.py`     | `ResearchAgent` — the LangGraph pipeline (business logic)      |
| `requirements.txt` | Python dependencies                                          |

## Notes

- The first run downloads the embedding model (~80 MB), the cross-encoder
  reranker (~80 MB), and the Chromium browser. No extra Python packages are
  needed beyond `requirements.txt` — the reranker runs on the existing
  `sentence-transformers`/`torch` stack.
- On macOS, `app.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`
  before importing torch/faiss to avoid a duplicate-OpenMP segfault.
- Async Playwright runs in a background thread with its own event loop so it
  plays nicely with Streamlit's synchronous execution model.
- Scraping is best-effort: pages that time out or block the headless browser are
  skipped, and the agent works with whatever it managed to collect. Recall is
  ultimately bounded by what the scraper can reach — the answer quality ceiling
  is the *sources*, not the retriever.
