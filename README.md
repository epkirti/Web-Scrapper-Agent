# Agentic Web Research Assistant

A Streamlit app wrapping the LangGraph web-scraping RAG agent from
`WebScrapper_Agents.ipynb`.

Given a question, the agent:

1. **Searches** the web (DuckDuckGo via `ddgs`)
2. **Scrapes** the result pages (Playwright / headless Chromium)
3. **Parses** the HTML to text (BeautifulSoup)
4. **Chunks** the text (LangChain text splitter)
5. **Embeds** the chunks (SentenceTransformers `all-MiniLM-L6-v2`)
6. **Indexes & retrieves** the most relevant chunks (FAISS)
7. **Evaluates** whether the context is sufficient (Groq LLM). If not, it
   **refines the query** and loops (up to *Max search rounds*); otherwise it
   **answers** strictly from the retrieved context.

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

- The first run downloads the embedding model (~80 MB) and the Chromium browser.
- Async Playwright runs in a background thread with its own event loop so it
  plays nicely with Streamlit's synchronous execution model.
- Scraping is best-effort: pages that time out or block the headless browser are
  skipped, and the agent works with whatever it managed to collect.
