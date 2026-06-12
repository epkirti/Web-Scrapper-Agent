"""Streamlit front-end for the agentic web-research RAG pipeline.

Run with:
    streamlit run app.py
"""

import os

# faiss and torch each bundle their own OpenMP runtime; on macOS loading both
# can segfault. These must be set BEFORE torch/faiss are imported (below).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Use the project-local HuggingFace cache so the pre-downloaded embedding /
# reranker models are found regardless of how the app is launched (e.g.
# `streamlit run app.py` directly, not only via run.ps1). Must be set BEFORE
# sentence_transformers / transformers are imported below.
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf-cache")
)
# Same for the Playwright browsers used by the Deep-dive scraper — they live in
# the project's .pw-browsers, so point Playwright there no matter how it's
# launched (otherwise Deep dive fails with "browser executable not found").
_PW_BROWSERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pw-browsers")
if os.path.isdir(_PW_BROWSERS):
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", _PW_BROWSERS)

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder

from scraper import ResearchAgent, ScraperConfig, STEP_LABELS
from google_ai_overview import fetch_ai_overview_sync
from news_map import (
    reverse_geocode, forward_geocode, fetch_area_news, summarize_news,
    fetch_weather, fetch_elevation, fetch_wikipedia_summary,
    web_search, summarize_topic,
)

# Persistent Chrome profile for the Quick-answer fetcher. Reusing it across runs
# builds up cookies/trust. Kept inside the project (the C: drive is full here).
CHROME_PROFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chrome-profile")

RAG_MODE = "Deep research (RAG)"
OVERVIEW_MODE = "Quick answer"
NEWS_MAP_MODE = "Area explorer (map)"

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
# Area Explorer page — search or tap a place, then pick what to learn about it
# --------------------------------------------------------------------------- #
DEFAULT_CENTER = [22.97, 78.65]  # India
DEFAULT_ZOOM = 4
CLICK_ZOOM = 11  # zoom level we fly to when a place is selected

# Knowledge categories handled generically (web search → LLM summary → sources).
# label -> the topic phrase used to search & prompt.
KNOWLEDGE_TOPICS = {
    "📜 History": "history",
    "💰 Economy": "economy and major industries",
    "🏖️ Tourism": "tourism, attractions and places to visit",
    "🎓 Education": "education, schools, colleges and universities",
    "👥 Demographics": "demographics, population, languages and religion",
    "🏛️ Government": "government, administration and local governance",
    "🚆 Transport": "transport, airport, railways and road connectivity",
    "🎭 Culture": "culture, traditions, festivals, cuisine and the arts",
    "🌱 Agriculture": "agriculture, main crops, farming and irrigation",
    "🏭 Industry": "industries, manufacturing and major businesses",
    "🌳 Environment": "environment, climate, biodiversity and conservation",
    "🏥 Healthcare": "healthcare, hospitals and medical facilities",
    "🔬 Science & Technology": "science, technology, research and innovation",
    "⚽ Sports": "sports, athletes and sporting facilities",
    "📈 Development": "development, infrastructure projects and economic growth",
}

# Dropdown order (News / Geography / Weather first, then the knowledge topics,
# then a general Overview).
CATEGORIES = (
    ["📰 News", "🌍 Geography", "🌦️ Weather"]
    + list(KNOWLEDGE_TOPICS)
    + ["📖 Overview"]
)


def _select_area(geo: dict, cur_zoom=None, news_hint: str = "") -> None:
    """Record the chosen place, clear cached category data, and queue a fly-to.

    ``news_hint`` is the user's raw search text (e.g. "Palasia Indore"). It is
    used as the NEWS subject so searching a specific locality gives that area's
    news, while geography/weather/etc. stay at the resolved city level. A map tap
    has no hint, so its news falls back to the resolved area (the city)."""
    st.session_state["area_selected"] = {
        "lat": geo["lat"], "lon": geo["lon"],
        "area": geo.get("area", ""), "detail": geo.get("detail", ""),
        "locality": geo.get("locality", ""), "state": geo.get("state", ""),
        "country": geo.get("country", ""), "address": geo.get("address", {}),
        "type": geo.get("type", ""), "category": geo.get("category", ""),
        "news_hint": (news_hint or "").strip(),
    }
    st.session_state["area_cache"] = {}  # new place -> drop old category results
    target_zoom = max(int(cur_zoom or DEFAULT_ZOOM), CLICK_ZOOM)
    # Persisted view = where the map is rebuilt from on plain reruns (category
    # switch, chat). A one-shot force_view actually flies the map there on select.
    st.session_state["area_view_center"] = [geo["lat"], geo["lon"]]
    st.session_state["area_view_zoom"] = target_zoom
    st.session_state["area_force_view"] = {
        "center": [geo["lat"], geo["lon"]], "zoom": target_zoom,
    }


def render_area_map(api_key: str, model: str, serper_api_key: str,
                    news_count: int, do_summary: bool) -> None:
    """Generic 'tap or search a place, then explore it' map.

    A place can be selected two ways — by typing in the search bar (forward
    geocode) or by tapping the map (reverse geocode). Either way it drops a
    marker, zooms in, and shows whichever category the user picks from the
    dropdown (News / Weather / Geography / Overview). Per-category results are
    cached so flipping the dropdown doesn't refetch. State lives in
    session_state because st_folium re-runs the script on every interaction."""
    import folium
    from streamlit_folium import st_folium

    st.caption("Search a place or tap the map, then choose what you want to know about it.")

    # ----- Search bar (forward geocode) ---------------------------------- #
    sc1, sc2 = st.columns([6, 1])
    with sc1:
        search_q = st.text_input(
            "Search a place", key="area_search_box", label_visibility="collapsed",
            placeholder="🔍 Search a place… e.g. Indore, Tokyo, Eiffel Tower",
        )
    with sc2:
        search_clicked = st.button("Search", use_container_width=True)

    q = (search_q or "").strip()
    # Fire on button OR Enter (Enter reruns with the same text -> dedupe via last).
    if q and (search_clicked or q != st.session_state.get("area_last_search", "")):
        st.session_state["area_last_search"] = q
        with st.spinner(f"Locating “{q}”…"):
            geo = forward_geocode(q)
        if geo.get("lat") is not None:
            # Use the typed query as the news subject so searching a specific
            # locality (e.g. "Palasia Indore") yields that area's news.
            _select_area(geo, news_hint=q)
            st.rerun()
        else:
            st.warning(f"Couldn't locate “{q}”. Try a more specific name.")

    # ----- Build the map ------------------------------------------------- #
    # Rebuild at the last-known view so reruns (category switches, etc.) don't
    # snap the map back to the default; force_view still overrides on selection.
    view_center = st.session_state.get("area_view_center", DEFAULT_CENTER)
    view_zoom = st.session_state.get("area_view_zoom", DEFAULT_ZOOM)
    fmap = folium.Map(location=view_center, zoom_start=view_zoom, tiles="OpenStreetMap")
    sel = st.session_state.get("area_selected")
    if sel:  # pin the currently selected place
        folium.Marker(
            [sel["lat"], sel["lon"]],
            tooltip=f"{sel.get('area') or 'Selected place'} ({sel['lat']:.5f}, {sel['lon']:.5f})",
            popup=folium.Popup(
                f"<b>{sel.get('area') or 'Selected place'}</b><br>"
                f"Lat: {sel['lat']:.5f}<br>Lon: {sel['lon']:.5f}",
                max_width=260,
            ),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(fmap)

    # One-shot fly-to: only set right after a place is selected, so the map moves
    # there exactly once. It's popped, so plain reruns don't re-apply it.
    force_view = st.session_state.pop("area_force_view", None)

    col_map, col_info = st.columns([3, 2])
    with col_map:
        # returned_objects=["last_clicked"] is the key fix: st_folium then re-runs
        # ONLY on a click — never on pan/zoom. So the user can freely zoom in and
        # the map will NOT snap back on a debounced rerun. center/zoom are passed
        # only on a deliberate selection (force_view) to fly the map there.
        map_state = st_folium(
            fmap,
            key="area_map",
            center=force_view["center"] if force_view else None,
            zoom=force_view["zoom"] if force_view else None,
            returned_objects=["last_clicked"],
            height=540,
            use_container_width=True,
        )

    # ----- Handle a fresh map tap (reverse geocode) ---------------------- #
    clicked = (map_state or {}).get("last_clicked")
    if clicked:
        # Dedupe against the last processed click so search/pan reruns don't
        # re-fire the old click (st_folium keeps last_clicked across reruns).
        lc = (round(clicked["lat"], 6), round(clicked["lng"], 6))
        if lc != st.session_state.get("area_last_click"):
            st.session_state["area_last_click"] = lc
            with st.spinner("Locating the tapped point…"):
                geo = reverse_geocode(clicked["lat"], clicked["lng"])
            geo["lat"], geo["lon"] = clicked["lat"], clicked["lng"]
            if not geo.get("area"):
                geo["area"] = f"{clicked['lat']:.3f}, {clicked['lng']:.3f}"
            _select_area(geo, cur_zoom=(map_state or {}).get("zoom"))
            st.rerun()

    # ----- Info panel ---------------------------------------------------- #
    with col_info:
        category = st.selectbox("Show me", CATEGORIES, key="area_category")

        sel = st.session_state.get("area_selected")
        if not sel:
            st.info("👆 Search a place or tap the map to begin.")
            return

        # What/where is selected.
        st.markdown(
            f"**📍 Coordinates** — Lat `{sel['lat']:.5f}` · Lon `{sel['lon']:.5f}`"
        )
        st.markdown(f"**🗺️ Location:** {sel.get('area') or 'Unknown place'}")
        if sel.get("detail"):
            st.caption(sel["detail"])
        st.divider()

        with st.spinner(f"Loading {category}…"):
            data = _get_category_data(
                category, sel, api_key, model, serper_api_key, news_count, do_summary
            )

        if category.endswith("News"):
            _render_news(data, sel, api_key, model, serper_api_key)
        elif "Weather" in category:
            _render_weather(data, sel)
        elif "Geography" in category:
            _render_geography(data, sel)
        elif category in KNOWLEDGE_TOPICS:
            _render_topic(category, data, sel, api_key, serper_api_key)
        else:  # Overview
            _render_overview(data, sel)

    # ----- Place-scoped chatbot (full width, below the map + info) ------- #
    # st.chat_input cannot live inside a column/expander, so it sits here at the
    # page body level. Only shown once a place is selected.
    sel = st.session_state.get("area_selected")
    if sel:
        _render_area_chat(sel, api_key, model, serper_api_key)


def _news_subject(sel: dict) -> str:
    """What the news is about: the typed search text if any (so a searched
    locality gives that area's news), otherwise the resolved area (the city)."""
    return (sel.get("news_hint") or "").strip() or sel.get("area", "")


def _get_category_data(category, sel, api_key, model, serper_api_key, news_count, do_summary):
    """Fetch (and cache per category) the data for the selected place."""
    cache = st.session_state.setdefault("area_cache", {})
    if category in cache:
        return cache[category]

    lat, lon = sel["lat"], sel["lon"]
    place = sel.get("area", "")
    if category.endswith("News"):
        subject = _news_subject(sel)
        items = fetch_area_news(subject, serper_api_key=serper_api_key, max_results=news_count)
        # If a specific searched locality returned little, widen to the city.
        if subject != place and place and len(items) < 2:
            city_items = fetch_area_news(place, serper_api_key=serper_api_key, max_results=news_count)
            if len(city_items) > len(items):
                items, subject = city_items, place
        summary = ""
        if do_summary and items and api_key.strip():
            summary = summarize_news(Groq(api_key=api_key.strip()), model.strip(), subject, items)
        data = {"items": items, "summary": summary, "subject": subject}
    elif "Weather" in category:
        data = fetch_weather(lat, lon)
    elif "Geography" in category:
        data = {"elevation": fetch_elevation(lat, lon)}
    elif category in KNOWLEDGE_TOPICS:
        topic = KNOWLEDGE_TOPICS[category]
        snippets = web_search(f"{place} {topic}", serper_api_key=serper_api_key, max_results=6)
        summary = ""
        if api_key.strip() and (snippets or place):
            wiki = fetch_wikipedia_summary(sel.get("locality") or place.split(",")[0])
            summary = summarize_topic(Groq(api_key=api_key.strip()), model.strip(),
                                      place, topic, snippets, wiki.get("extract", ""))
        data = {"summary": summary, "sources": snippets}
    else:  # Overview
        title = sel.get("locality") or place.split(",")[0]
        data = fetch_wikipedia_summary(title)
        if not data and sel.get("state"):  # fall back to the broader region
            data = fetch_wikipedia_summary(sel["state"])

    # Only cache results that actually carry content, so a transient network
    # failure (empty result) is retried next time rather than sticking.
    cacheable = bool(data) and (not isinstance(data, dict) or any(data.values()))
    # If a key is set but the AI summary/digest came back empty despite having
    # material (a transient LLM failure), don't cache — so re-viewing retries it.
    if api_key.strip() and not (data or {}).get("summary"):
        if category in KNOWLEDGE_TOPICS and (data or {}).get("sources"):
            cacheable = False
        elif category.endswith("News") and (data or {}).get("items"):
            cacheable = False
    if cacheable:
        cache[category] = data
    return data


def _render_news(data, sel, api_key, model, serper_api_key):
    subject = data.get("subject") or sel.get("area") or "this area"
    st.subheader(f"📰 News — {subject}")

    items = data.get("items") or []
    summary = data.get("summary", "")

    # Concise digest first: the consolidated "everything in one place" view.
    if summary:
        st.info(f"**📋 In short**\n\n{summary}")
    elif items and not api_key.strip():
        st.caption("🔑 Add a Groq API key in the sidebar for a concise digest of the headlines below.")

    if not items:
        st.warning("No news found for this area. Try a nearby point or a larger place.")
    else:
        # Headlines below the digest, collapsed by default once we have a digest.
        with st.expander(f"📰 All headlines ({len(items)})", expanded=not summary):
            for it in items:
                title, url = it.get("title", ""), it.get("url", "")
                st.markdown(f"**[{title}]({url})**" if url else f"**{title}**")
                meta = " · ".join(x for x in [it.get("source", ""), it.get("date", "")] if x)
                if meta:
                    st.caption(meta)
                if it.get("snippet"):
                    st.write(it["snippet"])
                st.divider()

    if api_key.strip() and st.button("🔬 Deep dive — cited summary of this area"):
        # Tuned for SPEED: one search round, few pages, short page timeout. The
        # full default pipeline (3 rounds × 10 pages × 15s) is what made it slow.
        config = ScraperConfig(
            model=model.strip(),
            serper_api_key=serper_api_key.strip(),
            max_searches=1,        # no re-query loop
            max_results=4,         # scrape only the top few pages
            page_timeout_ms=8000,  # don't wait on slow pages
            first_stage_k=15,
            pdf_max_pages=8,
        )
        emb = get_embedding_model(config.embedding_model_name)
        rer = get_reranker(config.reranker_model_name) if config.reranker_model_name else None
        agent = ResearchAgent(Groq(api_key=api_key.strip()), emb, config, reranker=rer)

        merged, failed = {}, False
        # Stream the steps live so progress is visible instead of one long spinner.
        with st.status("Running deep research… (~15–30s)", expanded=True) as status:
            try:
                for kind, payload in agent.stream_research(f"latest news about {subject}"):
                    if kind == "error":
                        failed = True
                        status.update(label="Deep dive failed", state="error")
                        break
                    node, node_state = next(iter(payload.items()))
                    merged.update(node_state)
                    status.write(STEP_LABELS.get(node, node))
                if not failed:
                    status.update(label="Done", state="complete")
            except Exception:  # graceful, like the main RAG flow
                failed = True
                status.update(label="Deep dive failed", state="error")

        if failed:
            st.error("Deep dive couldn't finish. Please try again.")
        else:
            st.markdown(merged.get("answer") or "_No answer was produced._")
            score = merged.get("confidence_score")
            if score is not None:
                st.caption(f"Confidence score {score:.2f}")


def _render_weather(data, sel):
    st.subheader(f"🌦️ Weather in {sel.get('area') or 'this area'}")
    if not data:
        st.warning("Weather is unavailable for this point right now.")
        return
    temp, feels = data.get("temp"), data.get("feels")
    lines = [f"**Condition:** {data.get('condition', '—')}"]
    if temp is not None:
        t = f"**Temperature:** {temp}°C"
        if feels is not None:
            t += f" (feels like {feels}°C)"
        lines.append(t)
    if data.get("humidity") is not None:
        lines.append(f"**Humidity:** {data['humidity']}%")
    if data.get("wind") is not None:
        lines.append(f"**Wind:** {data['wind']} km/h")
    if data.get("precip") is not None:
        lines.append(f"**Precipitation:** {data['precip']} mm")
    st.markdown("  \n".join(lines))
    st.caption("Current conditions · source: Open-Meteo")


def _render_geography(data, sel):
    st.subheader(f"🌍 Geography of {sel.get('area') or 'this area'}")
    addr = sel.get("address") or {}
    district = addr.get("state_district") or addr.get("county") or ""
    lines = [f"**Coordinates:** {sel['lat']:.5f}, {sel['lon']:.5f}"]
    if data.get("elevation") is not None:
        lines.append(f"**Elevation:** {data['elevation']:.0f} m above sea level")
    if sel.get("type"):
        lines.append(f"**Place type:** {sel['type'].replace('_', ' ')}")
    if sel.get("locality"):
        lines.append(f"**City / town:** {sel['locality']}")
    if district:
        lines.append(f"**District:** {district}")
    if sel.get("state"):
        lines.append(f"**State / region:** {sel['state']}")
    if sel.get("country"):
        lines.append(f"**Country:** {sel['country']}")
    st.markdown("  \n".join(lines))


def _render_topic(category, data, sel, api_key, serper_api_key):
    """Generic knowledge category (history, economy, tourism, …)."""
    st.subheader(f"{category} — {sel.get('area') or 'this area'}")
    summary = (data or {}).get("summary", "")
    sources = (data or {}).get("sources") or []

    if summary:
        # Best case: an AI-written summary from the sources.
        st.markdown(summary)
    elif sources:
        # No AI summary (no Groq key, or the model call failed/returned nothing) —
        # still show readable content by surfacing the source snippets themselves,
        # so the topic is never just a list of bare links.
        if not api_key.strip():
            st.caption("🔑 Add a Groq API key in the sidebar for a written summary. "
                       "Here's what the sources say:")
        else:
            st.caption("Here's what the sources say:")
        any_body = False
        for s in sources:
            body = (s.get("snippet") or "").strip()
            if body:
                any_body = True
                st.markdown(f"**{s.get('title', '')}**  \n{body}")
        if not any_body:
            st.info("The sources didn't include readable summaries. See the links below.")
    else:
        st.warning("Couldn't find information on this for this place. "
                   "Try a larger or more well-known place name.")

    if sources:
        st.divider()
        st.caption("Sources")
        for s in sources:
            t, u = s.get("title", ""), s.get("url", "")
            st.markdown(f"- [{t}]({u})" if u else f"- {t}")


def _render_overview(data, sel):
    st.subheader(f"📖 Overview of {sel.get('area') or 'this area'}")
    if not data or not data.get("extract"):
        st.info("No encyclopedic overview was found for this place.")
        return
    if data.get("thumb"):
        st.image(data["thumb"], width=220)
    st.write(data["extract"])
    if data.get("url"):
        st.markdown(f"[Read more on Wikipedia]({data['url']})")


# --------------------------------------------------------------------------- #
# Place-scoped chatbot — ask follow-up questions about the selected place
# --------------------------------------------------------------------------- #
def _area_chat_context(sel: dict, snippets: list | None = None, max_chars: int = 4500) -> str:
    """Ground the chatbot in the place's facts, anything already fetched for it
    (cached category results), and fresh web snippets for the current question."""
    parts = []

    facts = []
    for key, label in (("locality", "Locality"), ("state", "State/region"),
                       ("country", "Country"), ("type", "Place type")):
        if sel.get(key):
            facts.append(f"{label}: {sel[key]}")
    facts.append(f"Coordinates: {sel['lat']:.4f}, {sel['lon']:.4f}")
    parts.append("PLACE FACTS:\n" + "\n".join(facts))

    # Reuse whatever the user already loaded for this place (news, weather, …).
    for cat, d in (st.session_state.get("area_cache") or {}).items():
        if not isinstance(d, dict):
            continue
        bits = []
        if d.get("summary"):
            bits.append(d["summary"])
        if d.get("extract"):
            bits.append(d["extract"])
        if d.get("items"):
            heads = "; ".join(it.get("title", "") for it in d["items"][:6] if it.get("title"))
            if heads:
                bits.append("Recent headlines: " + heads)
        if d.get("condition"):  # weather payload
            bits.append(f"Current weather: {d.get('condition')}, {d.get('temp')}°C")
        if bits:
            parts.append(f"{cat}:\n" + "\n".join(bits))

    if snippets:
        web = "\n".join(
            f"- {s.get('title', '')}: {s.get('snippet', '')}".strip(" -:")
            for s in snippets if s.get("title") or s.get("snippet")
        )
        if web:
            parts.append("WEB RESULTS FOR THIS QUESTION:\n" + web)

    return "\n\n".join(parts)[:max_chars]


def _area_chat_reply(client, model: str, place: str, context: str, history: list) -> str:
    """A concise, accurate answer about the place, grounded in CONTEXT, falling
    back to general knowledge when the context doesn't cover the question."""
    system = (
        f"You are a knowledgeable, friendly guide answering questions about {place}. "
        "Use the CONTEXT below as your primary source when it is relevant. If the "
        "context does not cover the question, answer from reliable general knowledge, "
        "clearly and concisely. If you genuinely do not know, say so honestly rather "
        f"than guessing. Keep answers accurate and to the point.\n\nCONTEXT:\n{context}"
    )
    messages = [{"role": "system", "content": system}, *history[-8:]]
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.3, max_tokens=600,
    )
    return resp.choices[0].message.content or ""


def _render_area_chat(sel: dict, api_key: str, model: str, serper_api_key: str) -> None:
    place = sel.get("area") or "this place"
    st.divider()
    st.subheader(f"💬 Ask about {place}")
    st.caption("Ask follow-up questions — answers use what's been gathered here plus a quick web check.")

    # History is scoped to this place and resets when a new place is chosen.
    sig = f"{sel['lat']:.5f},{sel['lon']:.5f}"
    if st.session_state.get("area_chat_sig") != sig:
        st.session_state["area_chat_sig"] = sig
        st.session_state["area_chat"] = []

    for msg in st.session_state.get("area_chat", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Only show the input once a key is present — a persistent note otherwise
    # (a transient on-submit warning would just flash and vanish on the next rerun).
    if not api_key.strip():
        st.info("🔑 Add a Groq API key in the sidebar to chat about this place.")
        return

    user_q = st.chat_input(f"e.g. What is {place} famous for? Best time to visit?")
    if not user_q:
        return

    st.session_state["area_chat"].append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                snippets = web_search(f"{place} {user_q}", serper_api_key=serper_api_key, max_results=5)
                context = _area_chat_context(sel, snippets)
                reply = _area_chat_reply(
                    Groq(api_key=api_key.strip()), model.strip(), place, context,
                    st.session_state["area_chat"],
                )
            except Exception as exc:  # surfaced, never crashes the chat
                reply = f"Sorry, I couldn't answer that just now ({type(exc).__name__})."
        st.markdown(reply)

    st.session_state["area_chat"].append({"role": "assistant", "content": reply})


# --------------------------------------------------------------------------- #
# Sidebar — configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Mode",
        [RAG_MODE, OVERVIEW_MODE, NEWS_MAP_MODE],
        help=(
            "Deep research: searches multiple pages and answers with RAG.\n\n"
            "Quick answer: returns a short, direct answer.\n\n"
            "Area explorer: search or tap a place to see its news, weather, "
            "geography, and overview."
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
    elif mode == OVERVIEW_MODE:
        with st.expander("Advanced"):
            overview_timeout_ms = st.number_input(
                "Answer wait timeout (ms)", 10000, 60000, 25000, step=1000,
            )
    else:  # NEWS_MAP_MODE — Area explorer
        serper_api_key = st.text_input(
            "Serper API key (Google News)",
            type="password",
            value=os.getenv("SERPER_API_KEY", ""),
            help="Free key at https://serper.dev — gives Google News. Leave blank to use DuckDuckGo.",
        )
        with st.expander("News options"):
            news_count = st.slider("Headlines per area", 3, 20, 8)
            do_summary = st.checkbox("AI summary of the area's news", value=True)
        st.divider()
        st.caption(
            "search / tap a place → news · weather · geography · overview\n\n"
            "Weather, geography & overview are free (no key needed)."
        )


# --------------------------------------------------------------------------- #
# Main — query & run
# --------------------------------------------------------------------------- #
st.title("🔎 Agentic Web Research Assistant")
st.caption("LangGraph agent that searches the web, scrapes pages, and answers with RAG over what it finds.")

# Area explorer mode is fully self-contained (its own map + search + click handling).
if mode == NEWS_MAP_MODE:
    render_area_map(api_key, model, serper_api_key, news_count, do_summary)
    st.stop()

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
                headless=True,         # fully invisible: no window, no taskbar
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
