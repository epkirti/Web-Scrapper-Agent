"""Geo Explorer for classes 6-10 — pick a place by SEARCH or by TAPPING the map
(one mode at a time), get a grade-appropriate, cited geography explainer.

Run with:
    streamlit run geo_app.py     (NOT app.py — app.py is the text-only research UI)

Mode is exclusive: in "Search" mode the map tap is disabled; in "Tap" mode the
search bar is disabled. Either way the chosen location is turned into a
class-tailored question and answered by the existing agentic web-research
pipeline (scraper.py), unchanged. The map is served with streamlit-folium and
the result marker carries a tappable flyer (popup).
"""

import os

# faiss and torch each bundle their own OpenMP runtime; set these BEFORE they are
# imported (transitively, below) to avoid a segfault when both load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import html
import json
import re

import folium
import streamlit as st
from streamlit_folium import st_folium
from groq import Groq
from sentence_transformers import CrossEncoder, SentenceTransformer
from geopy.geocoders import Nominatim

from scraper import ResearchAgent, ScraperConfig, STEP_LABELS
from geo_education import PlaceInfo, reverse_geocode, build_student_query

st.set_page_config(page_title="Geo Explorer for Students", page_icon="🗺️", layout="wide")

GROQ_MODEL = "llama-3.3-70b-versatile"


# --------------------------------------------------------------------------- #
# Cached heavy resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


@st.cache_resource(show_spinner="Loading reranker model…")
def get_reranker(name: str) -> CrossEncoder:
    return CrossEncoder(name, max_length=512)


@st.cache_resource
def get_forward_geocoder() -> Nominatim:
    return Nominatim(user_agent="geo-edu-classroom")


def _chat_context(merged: dict, max_chars: int = 4000) -> str:
    """Ground the chatbot in what the pipeline already gathered for this place."""
    parts = []
    answer = (merged.get("answer") or "").strip()
    if answer:
        parts.append("SUMMARY:\n" + answer)
    chunks = merged.get("retrieved_chunks") or []
    if chunks:
        parts.append("SOURCE NOTES:\n" + "\n\n".join(str(c) for c in chunks))
    return "\n\n".join(parts)[:max_chars]


def _chat_reply(client, grade: int, place_name: str, context: str, history: list) -> str:
    """A grade-appropriate, place-scoped answer grounded in the gathered context."""
    system = (
        f"You are a friendly, encouraging geography teacher helping a class {grade} "
        f"student in India who is learning about {place_name}. Use the CONTEXT below "
        f"as your primary source when it is relevant. If the context does not cover "
        f"the question, answer from your own reliable general geography knowledge "
        f"(for example, why a place has a certain climate, how rivers or landforms "
        f"work) — clearly, accurately, and in simple language for a class {grade} "
        f"student. Only say you are unsure if you genuinely do not know the answer. "
        f"Prefer to stay on geography topics (places, climate, landforms, rivers, "
        f"soil, water, maps, environment). Avoid politics, violence, religious "
        f"disputes, and any casualty or disaster-death details.\n\nCONTEXT:\n{context}"
    )
    messages = [{"role": "system", "content": system}, *history[-8:]]
    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.3, max_tokens=600,
    )
    return resp.choices[0].message.content or ""


def _generate_quiz(
    client, grade: int, place_name: str, context: str,
    n: int = 3, round_no: int = 1, avoid: list | None = None,
) -> list:
    """Make `n` grade-appropriate MCQs about the place, grounded in the context.

    ``round_no`` + ``avoid`` (questions already asked here) push the model to
    produce a *different* set each time the student requests a new quiz.
    """
    avoid_txt = ""
    if avoid:
        joined = "\n".join(f"- {a}" for a in avoid[-12:])
        avoid_txt = (
            "\nThese questions were already asked — do NOT repeat or lightly reword "
            f"them; cover different facts and layers instead:\n{joined}\n"
        )
    prompt = (
        f"Create {n} NEW multiple-choice geography questions (variation set #{round_no}) "
        f"for a class {grade} student about {place_name}, using ONLY the CONTEXT. Vary "
        f"the focus across different layers (climate, terrain, soil, water, natural "
        f"disturbances) and different facts. Each question must have exactly 4 options "
        f"with ONE correct answer. Use simple language for a class {grade} student. "
        f"Avoid politics, violence, and casualty details.{avoid_txt}\n"
        'Return ONLY JSON of this shape: '
        '{"questions":[{"question":"...","options":["a","b","c","d"],"answer":0,'
        '"why":"one-line reason"}]}\n\nCONTEXT:\n' + context
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0.9, top_p=0.95, max_tokens=900, response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}

    quiz = []
    for q in data.get("questions", []):
        opts = q.get("options") or []
        ans = q.get("answer")
        if (
            isinstance(q.get("question"), str)
            and isinstance(opts, list) and len(opts) >= 2
            and isinstance(ans, int) and 0 <= ans < len(opts)
        ):
            quiz.append({
                "question": q["question"],
                "options": [str(o) for o in opts],
                "answer": ans,
                "why": str(q.get("why", "")),
            })
    return quiz


def _flyer_html(place_name: str, grade: int, answer: str | None) -> str:
    """Build the marker popup ('flyer') — place, class, and a short snippet."""
    snippet = (answer or "").replace("*", "").replace("\n", " ").strip()
    if len(snippet) > 240:
        snippet = snippet[:240].rsplit(" ", 1)[0] + "…"
    body = f"<br>{html.escape(snippet)}" if snippet else "<br><i>Researching…</i>"
    return (
        f"<div style='font-size:13px;max-width:300px'>"
        f"<b>{html.escape(place_name)}</b><br>"
        f"<span style='color:#555'>Class {grade} geography</span>{body}</div>"
    )


# --------------------------------------------------------------------------- #
# Sidebar — keys & research depth
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""),
        help="Get one at https://console.groq.com/keys",
    )
    serper_api_key = st.text_input(
        "Serper API key (optional)", type="password", value=os.getenv("SERPER_API_KEY", ""),
        help="Free key at https://serper.dev for Google results. Blank → DuckDuckGo.",
    )
    language = st.selectbox("Place-name language", ["en", "hi"], index=0)
    with st.expander("Research depth"):
        max_searches = st.slider("Max research rounds", 1, 6, 3)
        verify_answer = st.checkbox("Fact-check pass", value=True)
        rerank = st.checkbox("Cross-encoder reranking", value=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
st.session_state.setdefault("center", [22.5, 79.0])   # India
st.session_state.setdefault("zoom", 4)
st.session_state.setdefault("marker", None)           # {lat, lon, label, popup}
st.session_state.setdefault("last_sig", None)         # last researched target
st.session_state.setdefault("seen_click_sig", None)   # last map click we reacted to
st.session_state.setdefault("result", None)           # {place, grade, question, merged}


# --------------------------------------------------------------------------- #
# Top controls — exclusive mode + search bar + class input field
# --------------------------------------------------------------------------- #
st.title("🗺️ Geo Explorer — geography for classes 6–10")

mode = st.radio(
    "How do you want to choose a place?",
    ["🔍 Search by name", "🗺️ Tap on the map"],
    horizontal=True,
    help="Pick one. The other input is disabled until you switch back.",
)
is_search = mode.startswith("🔍")

c_search, c_class, c_btn = st.columns([5, 1.4, 1.4], vertical_alignment="bottom")
search_text = c_search.text_input(
    "🔍 Search a place or geography topic",
    placeholder="e.g. Bhopal  ·  Western Ghats  ·  Sundarbans mangroves",
    disabled=not is_search,
)
grade = c_class.number_input("🎓 Class (6–10)", min_value=6, max_value=10, value=8, step=1)
search = c_btn.button(
    "Search", type="primary", use_container_width=True, disabled=not is_search,
)

st.caption(
    "🔍 **Search mode:** type a place above. &nbsp;&nbsp; 🗺️ **Tap mode:** click the map. "
    "Switch modes with the toggle — only one is active at a time."
)


# --------------------------------------------------------------------------- #
# Resolve a research target from the SEARCH bar (only in search mode; runs
# before the map so it can recenter in the same rerun).
# --------------------------------------------------------------------------- #
target = None  # {"sig", "lat", "lon", "place"|None}

if is_search and search and search_text.strip():
    q = search_text.strip()
    try:
        loc = get_forward_geocoder().geocode(q, language="en", timeout=10)
    except Exception:
        loc = None
    if loc is not None:
        st.session_state["center"] = [loc.latitude, loc.longitude]
        st.session_state["zoom"] = 9
        place = PlaceInfo(lat=loc.latitude, lon=loc.longitude, name=loc.address, raw=loc.raw or {})
        st.session_state["marker"] = {
            "lat": loc.latitude, "lon": loc.longitude, "label": loc.address,
            "popup": _flyer_html(loc.address, int(grade), None),
        }
        target = {"sig": f"q:{loc.latitude:.5f},{loc.longitude:.5f}", "place": place}
    else:
        # Not a mappable place (e.g. a topic) — research the raw text by name.
        st.session_state["marker"] = None
        target = {"sig": f"t:{q.lower()}", "place": PlaceInfo(lat=0.0, lon=0.0, name=q)}


# --------------------------------------------------------------------------- #
# Map (streamlit-folium) — tap to pick a point (only in tap mode)
# --------------------------------------------------------------------------- #
fmap = folium.Map(
    location=st.session_state["center"], zoom_start=st.session_state["zoom"], control_scale=True,
)
if not is_search:
    fmap.add_child(folium.LatLngPopup())  # show lat/lon on click while tapping

mk = st.session_state["marker"]
if mk:
    folium.Marker(
        [mk["lat"], mk["lon"]],
        tooltip="Tap for details",
        popup=folium.Popup(mk["popup"], max_width=320),
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(fmap)

map_state = st_folium(
    fmap, height=520, use_container_width=True, returned_objects=["last_clicked"],
)

# A map click becomes the target ONLY in tap mode and ONLY if it's a new click.
clicked = (map_state or {}).get("last_clicked")
if (not is_search) and clicked:
    lat, lon = float(clicked["lat"]), float(clicked["lng"])
    sig = f"c:{lat:.5f},{lon:.5f}"
    if sig != st.session_state["seen_click_sig"]:
        st.session_state["seen_click_sig"] = sig
        st.session_state["marker"] = {
            "lat": lat, "lon": lon, "label": "Selected point",
            "popup": _flyer_html(f"{lat:.3f}, {lon:.3f}", int(grade), None),
        }
        target = {"sig": sig, "lat": lat, "lon": lon, "place": None}


# --------------------------------------------------------------------------- #
# Research the target (only when it's new), cache result, enrich the flyer.
# --------------------------------------------------------------------------- #
if target and target["sig"] != st.session_state["last_sig"]:
    if not api_key.strip():
        st.error("Please enter your Groq API key in the sidebar to run research.")
    else:
        st.session_state["last_sig"] = target["sig"]

        place = target["place"]
        if place is None:  # map tap → resolve the place name
            with st.spinner("Finding this place…"):
                place = reverse_geocode(target["lat"], target["lon"], language=language)

        question = build_student_query(place, int(grade))

        config = ScraperConfig(
            model=GROQ_MODEL,
            serper_api_key=serper_api_key.strip(),
            max_searches=int(max_searches),
            rerank=bool(rerank),
            verify_answer=bool(verify_answer),
        )
        embedding_model = get_embedding_model(config.embedding_model_name)
        reranker = get_reranker(config.rerank_model_name) if config.rerank else None
        agent = ResearchAgent(Groq(api_key=api_key.strip()), embedding_model, config, reranker=reranker)

        merged: dict = {}
        with st.status(f"Researching {place.name}…", expanded=True) as status:
            for kind, payload in agent.stream_research(question):
                if kind == "error":
                    status.update(label="Research failed", state="error")
                    st.exception(payload)
                    st.stop()
                node, node_state = next(iter(payload.items()))
                merged.update(node_state)
                st.write(STEP_LABELS.get(node, node))
            status.update(label="Done", state="complete")

        st.session_state["result"] = {
            "place": place, "grade": int(grade), "question": question, "merged": merged,
        }
        # Enrich the marker flyer with a snippet, then rerun so it shows on the map.
        if not getattr(place, "is_water", False) or (place.lat or place.lon):
            st.session_state["marker"] = {
                "lat": place.lat, "lon": place.lon, "label": place.name,
                "popup": _flyer_html(place.name, int(grade), merged.get("answer")),
            }
        st.rerun()


# --------------------------------------------------------------------------- #
# Render the latest result (persists across map pans/zooms and mode switches)
# --------------------------------------------------------------------------- #
res = st.session_state.get("result")
if not res:
    st.info("👆 Pick a mode above, then search a place or tap the map.")
    st.stop()

place, merged = res["place"], res["merged"]
st.subheader(f"📚 {place.name} — for Class {res['grade']}")
if getattr(place, "is_water", False):
    st.caption("This point is on water / an unnamed area — researching the surrounding region.")

confidence = (merged.get("confidence") or "").lower()
if confidence == "high":
    st.success("🟢 High confidence — corroborated by multiple independent sources.")
elif confidence == "medium":
    st.warning("🟡 Medium confidence — supported, but mostly single-source.")

st.markdown(merged.get("answer") or "_No answer was produced._")

urls = merged.get("urls") or []
with st.expander(f"🔗 Sources ({len(urls)})"):
    if urls:
        for u in urls:
            st.markdown(f"- [{u}]({u})")
    else:
        st.write("No sources were scraped.")

with st.expander("🧪 The exact question asked (grade-tailored)"):
    st.write(res["question"])


# --------------------------------------------------------------------------- #
# 📝 Quiz — grade-appropriate MCQs grounded in the researched content.
# Scoped to this place; resets when a new place is chosen.
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("📝 Test yourself")

if st.session_state.get("quiz_sig") != st.session_state["last_sig"]:
    st.session_state["quiz_sig"] = st.session_state["last_sig"]
    st.session_state["quiz"] = None       # None = not generated yet; [] = none available
    st.session_state["quiz_round"] = 1     # bumps each "New quiz" for fresh questions
    st.session_state["quiz_seen"] = []     # question texts already asked for this place

if st.session_state.get("quiz") is None:
    if st.button(f"🎲 Generate a Class {res['grade']} quiz on {place.name}"):
        if not api_key.strip():
            st.error("Please enter your Groq API key in the sidebar to generate a quiz.")
        else:
            with st.spinner("Writing quiz questions…"):
                try:
                    quiz = _generate_quiz(
                        Groq(api_key=api_key.strip()), res["grade"], place.name,
                        _chat_context(merged),
                        round_no=st.session_state.get("quiz_round", 1),
                        avoid=st.session_state.get("quiz_seen", []),
                    )
                    st.session_state["quiz"] = quiz
                    st.session_state["quiz_seen"] = (
                        st.session_state.get("quiz_seen", []) + [q["question"] for q in quiz]
                    )
                except Exception as exc:
                    st.session_state["quiz"] = []
                    st.error(f"Couldn't generate a quiz ({type(exc).__name__}).")
            st.rerun()
else:
    quiz = st.session_state["quiz"]
    if not quiz:
        st.info("No quiz could be made from the gathered information yet.")
        if st.button("Try again"):
            st.session_state["quiz"] = None
            st.rerun()
    else:
        rnd = st.session_state.get("quiz_round", 1)
        with st.form(f"quiz_form_{rnd}"):
            picks = []
            for i, q in enumerate(quiz):
                st.markdown(f"**Q{i + 1}. {q['question']}**")
                picks.append(st.radio(
                    "Choose one:", q["options"], index=None,
                    key=f"quiz_{st.session_state['last_sig']}_{rnd}_{i}",
                    label_visibility="collapsed",
                ))
            submitted = st.form_submit_button("✅ Check my answers", type="primary")

        if submitted:
            score = 0
            for i, (q, pick) in enumerate(zip(quiz, picks)):
                correct = q["options"][q["answer"]]
                if pick == correct:
                    score += 1
                    st.success(f"Q{i + 1}: ✅ Correct — **{correct}**")
                else:
                    st.error(f"Q{i + 1}: ❌ You chose: {pick or '—'} · Correct: **{correct}**")
                if q.get("why"):
                    st.caption(f"Why: {q['why']}")
            st.markdown(f"### 🏆 Your score: {score}/{len(quiz)}")

        if st.button("🔄 New quiz"):
            st.session_state["quiz_round"] = rnd + 1
            st.session_state["quiz"] = None
            st.rerun()


# --------------------------------------------------------------------------- #
# 💬 Follow-up chatbot — students ask about the place they just researched.
# History is scoped to this place and resets when a new place is chosen.
# --------------------------------------------------------------------------- #
st.divider()
st.subheader(f"💬 Ask about {place.name}")

if st.session_state.get("chat_sig") != st.session_state["last_sig"]:
    st.session_state["chat_sig"] = st.session_state["last_sig"]
    st.session_state["chat"] = []  # [{role, content}], fresh per place

for msg in st.session_state.get("chat", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input(f"e.g. Which river flows through {place.name}? Why is it important?")
if user_q:
    if not api_key.strip():
        st.error("Please enter your Groq API key in the sidebar to chat.")
        st.stop()

    st.session_state["chat"].append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = _chat_reply(
                    Groq(api_key=api_key.strip()),
                    res["grade"],
                    place.name,
                    _chat_context(merged),
                    st.session_state["chat"],
                )
            except Exception as exc:  # surfaced instead of crashing the chat
                reply = f"Sorry, I couldn't answer that just now ({type(exc).__name__})."
        st.markdown(reply)

    st.session_state["chat"].append({"role": "assistant", "content": reply})
