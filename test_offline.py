"""Offline integration test for scraper.py — no network, no torch, no real LLM."""
import hashlib
import re
import sys

import numpy as np

import scraper


# ----------------------------- fakes ---------------------------------------- #
class FakeEmbedder:
    """Deterministic 32-dim vectors seeded from the text hash."""

    def encode(self, texts, normalize_embeddings=True, **kw):
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(32).astype("float32")
            v /= (np.linalg.norm(v) + 1e-9)
            out.append(v)
        return np.vstack(out)


class FakeReranker:
    """Scores by word overlap with the query."""

    def predict(self, pairs, **kw):
        scores = []
        for q, passage in pairs:
            qs = set(re.findall(r"\w+", q.lower()))
            ps = set(re.findall(r"\w+", passage.lower()))
            scores.append(len(qs & ps) / (len(qs) + 1))
        return np.array(scores)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeGroq:
    """Routes on system-prompt keywords; stateful to test both loops."""

    def __init__(self):
        self.eval_calls = 0
        self.verify_calls = 0
        self.calls = []

        class _Completions:
            def __init__(s_inner, outer):
                s_inner.outer = outer

            def create(s_inner, **kwargs):
                return s_inner.outer._create(**kwargs)

        class _Chat:
            def __init__(s_inner, outer):
                s_inner.completions = _Completions(outer)

        self.chat = _Chat(self)

    def _create(self, **kwargs):
        system = next((m["content"] for m in kwargs["messages"] if m["role"] == "system"), "")
        self.calls.append(system[:40])
        if "research planner" in system.lower() or "planner" in system.lower():
            return FakeResponse(
                '{"queries": ["mp wheat production 2024 tonnes", '
                '"madhya pradesh wheat output official statistics", '
                '"india wheat production by state 2024"]}'
            )
        if "auditor" in system.lower():
            self.eval_calls += 1
            if self.eval_calls == 1:  # force one research loop
                return FakeResponse(
                    '{"enough_info": false, "missing": "exact 2024 tonnage", '
                    '"followup_query": "MP wheat tonnes 2024"}'
                )
            return FakeResponse('{"enough_info": true, "missing": ""}')
        if "fact-checker" in system.lower():
            self.verify_calls += 1
            if self.verify_calls == 1:  # force one verify loop
                return FakeResponse(
                    '{"verdict": "fail", "unsupported_claims": ["MP exports 5 MT"], '
                    '"single_source_claims": [], "confidence": "low", "reason": "export claim unbacked"}'
                )
            return FakeResponse(
                '{"verdict": "pass", "unsupported_claims": [], '
                '"single_source_claims": ["second-largest producer"], '
                '"confidence": "high", "reason": "corroborated"}'
            )
        if "repair" in system.lower():
            return FakeResponse("Repaired answer with only supported claims [1].")
        # analyst → final answer
        return FakeResponse(
            "Madhya Pradesh produced 22.3 million tonnes of wheat in 2023-24 [1,2]. "
            "It is the second-largest wheat-producing state in India [2]."
        )


PAGES = {
    "https://agri.example.gov/wheat": (
        "Wheat production statistics. Madhya Pradesh produced 22.3 million tonnes "
        "of wheat in the 2023-24 season according to the agriculture ministry. "
        "This makes it a leading producer. " * 6
    ),
    "https://stats.example.org/india-wheat": (
        "India wheat output by state. Madhya Pradesh is the second-largest "
        "wheat-producing state with 22.3 million tonnes in 2023-24, behind "
        "Uttar Pradesh. Punjab ranks third. " * 6
    ),
    "https://news.example.com/mp-harvest": (
        "MP harvest news. Farmers in Madhya Pradesh reported a strong wheat "
        "harvest this rabi season with procurement at record levels. " * 6
    ),
}


def fake_ddg(query, n):
    return [
        {"title": f"Result for {u.rsplit('/', 1)[-1]}", "url": u,
         "snippet": PAGES[u][:150], "date": ""}
        for u in PAGES
    ]


def fake_fetch_one(self, client, url):
    text = PAGES.get(url)
    if text is None:
        raise RuntimeError("offline test: unknown URL " + url)
    return {"key": url, "url": url, "title": url.rsplit("/", 1)[-1], "date": "2024-05-01",
            "text": text, "kind": "page"}


def fake_playwright(urls, timeout_ms):
    return {}


# ----------------------------- patch & run ----------------------------------- #
scraper.ddg_search = fake_ddg
scraper.serper_search = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no serper in test"))
scraper.ResearchAgent._fetch_one = fake_fetch_one
scraper.run_playwright_in_thread = fake_playwright

cfg = scraper.ScraperConfig(max_searches=3, max_results=3, queries_per_round=3,
                            chunk_size=300, chunk_overlap=50, top_k=4, retrieve_k=10,
                            rerank=True, verify_answer=True, use_search_snippets=True)
agent = scraper.ResearchAgent(FakeGroq(), FakeEmbedder(), cfg, reranker=FakeReranker())

seq, merged = [], {}
for kind, payload in agent.stream_research("How much wheat does Madhya Pradesh produce?"):
    assert kind == "update", f"pipeline raised: {payload!r}"
    node, delta = next(iter(payload.items()))
    seq.append(node)
    merged.update(delta)
    print(f"  {node:9s} -> {sorted(delta.keys())}")

print("\nSEQUENCE:", " → ".join(seq))

# --- assertions --------------------------------------------------------------
assert seq[0] == "plan" and seq.count("plan") == 3, "expected 3 plan rounds (eval loop + verify loop)"
assert seq[-1] == "finalize"
assert "verify" in seq and seq.count("verify") == 2, "expected verify fail→pass"
for required in ("search", "scrape", "index", "retrieve", "evaluate", "answer"):
    assert required in seq, f"missing node {required}"

answer = merged["answer"]
assert "22.3 million tonnes" in answer, "answer lost the key figure"
assert "**Confidence:**" in answer and "🟢" in answer, "confidence badge missing"
assert "**Sources**" in answer, "sources list missing"
assert re.search(r"[\d, ]+\. \[.*\]\(https?://", answer), "sources not linked"
_src_block = answer.split("**Sources**")[1]
assert _src_block.count("stats.example.org/india-wheat") == 1, "duplicate URL rows in sources"
assert merged["confidence"] == "high"
assert merged["verification"]["verdict"] == "pass"
assert merged["sources"], "sources metadata missing"
assert merged["n_chunks"] > 0 and merged["retrieved_chunks"], "index/retrieval empty"
assert merged["search_provider"] == "DuckDuckGo"

print("\n----- FINAL ANSWER -----\n")
print(answer)
print("\nALL ASSERTIONS PASSED ✔")
