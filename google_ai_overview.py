"""Fetch Google's "AI Overview" block by driving a real browser.

Strategy that actually works for personal-volume use:
  * Use a PERSISTENT Chrome profile (real cookies/session) so Google sees a
    trusted browser, not a fresh headless bot.
  * Run headful by default (headless is trivially detected by Google).
  * Apply light stealth (hide navigator.webdriver, real UA, locale, viewport).
  * Handle the EU/consent interstitial, then wait for the AI Overview block and
    click its "Show more" so the full text is in the DOM before extracting.

Reality check: Google changes its markup and bot defenses constantly. There is
NO selector or trick that works 100% forever. If this returns nothing, Google
either didn't show an AI Overview for that query or served a CAPTCHA -- inspect
the screenshot it saves (debug=True) and adjust the selectors below.
"""

from __future__ import annotations

import asyncio
from urllib.parse import quote_plus

from playwright.async_api import async_playwright

# Looks like a normal up-to-date desktop Chrome on Windows.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# JS injected before any page script runs, to strip the obvious automation tells.
_STEALTH = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});
window.chrome = window.chrome || {runtime: {}, app: {}, csi: () => {}, loadTimes: () => {}};
// Pretend permissions behave like a normal browser (headless gives itself away here).
const _q = navigator.permissions && navigator.permissions.query;
if (_q) navigator.permissions.query = (p) =>
  p && p.name === 'notifications'
    ? Promise.resolve({state: Notification.permission})
    : _q(p);
// WebGL vendor/renderer strings of a real GPU, not SwiftShader (a headless tell).
const _gp = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(p) {
  if (p === 37445) return 'Intel Inc.';
  if (p === 37446) return 'Intel Iris OpenGL Engine';
  return _gp.apply(this, [p]);
};
"""


async def _dismiss_consent(page) -> None:
    """Click through Google's cookie/consent wall if it appears."""
    for label in ("Accept all", "I agree", "Reject all", "Accept the use"):
        try:
            btn = page.get_by_role("button", name=label)
            if await btn.count():
                await btn.first.click(timeout=2000)
                await page.wait_for_timeout(800)
                return
        except Exception:
            pass


async def _expand_overview(page) -> None:
    """Click the AI Overview 'Show more' so the full text is rendered.

    It is not always a real <button>, so try by-role AND by-text.
    """
    for label in ("Show more", "Show all"):
        for getter in (
            lambda l=label: page.get_by_role("button", name=l),
            lambda l=label: page.get_by_text(l, exact=True),
        ):
            try:
                el = getter()
                if await el.count():
                    await el.first.click(timeout=2000)
                    await page.wait_for_timeout(1500)
                    return
            except Exception:
                pass


# Runs in the page: find the "AI Overview" label, walk up to the content
# container (first ancestor whose text is substantially longer than the label),
# and return its text + outbound links. Class names are randomized, so we rely
# on this structural walk rather than CSS selectors.
_EXTRACT_JS = r"""
() => {
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n, label = null;
  while ((n = w.nextNode())) {
    if (n.textContent.trim() === 'AI Overview') { label = n.parentElement; break; }
  }
  if (!label) return {text: '', links: []};
  let el = label, container = null;
  for (let i = 0; i < 12 && el; i++, el = el.parentElement) {
    if ((el.innerText || '').trim().length > 200) { container = el; break; }
  }
  if (!container) return {text: '', links: []};
  const text = container.innerText.trim();
  const links = Array.from(container.querySelectorAll("a[href^='http']"))
                     .map(a => a.href);
  return {text, links};
}
"""


async def fetch_ai_overview(
    query: str,
    user_data_dir: str | None = None,
    headless: bool = False,
    hidden: bool = False,
    region: str = "in",
    channel: str | None = "chrome",
    timeout_ms: int = 20000,
    debug: bool = False,
) -> dict:
    """Return {'query', 'text', 'links', 'unavailable'} from Google's AI Overview.

    user_data_dir: path to a Chrome profile dir to reuse (recommended). If None,
                   a throwaway context is used and you'll get CAPTCHA'd quickly.
    headless:      true headless -- invisible but the MOST detectable; avoid.
    hidden:        run a REAL (headful) browser positioned off-screen, so it is
                   invisible to the user yet not flagged as headless. Preferred
                   way to hide the window. Ignored if headless=True.
    region:        Google 'gl' country code (default 'in' = India, matching the
                   user's real location -- AI Overview availability is regional).
    channel:       browser channel; 'chrome' uses your installed Google Chrome
                   (Google trusts it far more than bundled Chromium). Falls back
                   to Chromium automatically if Chrome isn't installed.
    """
    url = f"https://www.google.com/search?q={quote_plus(query)}&hl=en&gl={region}"

    args = ["--disable-blink-features=AutomationControlled", "--no-first-run"]
    if hidden and not headless:
        # Real browser, but parked far off any monitor so the user never sees it.
        args += ["--window-position=-32000,-32000", "--window-size=1366,900"]

    async def _open(p, use_channel):
        launch_kwargs = dict(headless=headless, args=args)
        if use_channel:
            launch_kwargs["channel"] = use_channel
        if user_data_dir:
            return await p.chromium.launch_persistent_context(
                user_data_dir, locale="en-US", user_agent=_UA,
                viewport={"width": 1366, "height": 900}, **launch_kwargs,
            )
        browser = await p.chromium.launch(**launch_kwargs)
        return await browser.new_context(
            locale="en-US", user_agent=_UA, viewport={"width": 1366, "height": 900},
        )

    async with async_playwright() as p:
        try:
            ctx = await _open(p, channel)
        except Exception:
            # Chrome not installed / channel unavailable -> bundled Chromium.
            ctx = await _open(p, None)
        await ctx.add_init_script(_STEALTH)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        await _dismiss_consent(page)

        # The overview loads async and STREAMS IN. Poll the structural extractor
        # until the text stops growing, Google VISIBLY says none is available, or
        # we time out. NOTE: Google ships a *hidden* "not available" fallback span
        # even while it is still generating ("Thinking..."), so we only treat it
        # as unavailable when that span is actually visible -- otherwise we'd give
        # up while the overview is still streaming in.
        text, links, unavailable = "", [], False
        expanded = False
        prev, stable, waited = -1, 0, 0
        while waited < timeout_ms:
            na = page.get_by_text("An AI Overview is not available", exact=False)
            if await na.count() and await na.first.is_visible():
                unavailable = True
                break

            res = await page.evaluate(_EXTRACT_JS)
            cur = len(res["text"])

            # Once the block exists, click its "Show more" a single time.
            if cur > 0 and not expanded:
                await _expand_overview(page)
                expanded = True

            if cur == prev and cur > 0:
                stable += 1
                if stable >= 2:            # unchanged across two polls -> done
                    text, links = res["text"], res["links"]
                    break
            elif cur > 0:
                text, links = res["text"], res["links"]
            else:
                stable = 0
            if debug:
                print(f"[dbg] poll {waited}ms: text len={cur}")
            prev = cur
            await page.wait_for_timeout(700)
            waited += 700

        if debug:
            await page.screenshot(path="ai_overview_debug.png", full_page=True)
            print(f"[dbg] unavailable={unavailable}")

        await ctx.close()

    # De-dupe links, drop google's own.
    seen, clean = set(), []
    for u in links:
        if "google.com" in u or u in seen:
            continue
        seen.add(u)
        clean.append(u)

    return {"query": query, "text": text, "links": clean, "unavailable": unavailable}


async def warmup_login(user_data_dir: str) -> None:
    """Open the persistent profile to Google in a VISIBLE window and wait for you
    to sign in / solve any CAPTCHA. Google gates the AI Overview behind a trusted
    (usually logged-in) session, so doing this once makes the profile much more
    likely to get overviews afterward. Cookies are saved into user_data_dir.
    """
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir, headless=False, locale="en-US", user_agent=_UA,
            viewport={"width": 1366, "height": 900},
            args=["--disable-blink-features=AutomationControlled", "--no-first-run"],
        )
        await ctx.add_init_script(_STEALTH)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto("https://www.google.com/", wait_until="domcontentloaded")
        print("\n>>> A Chrome window opened. Sign in to your Google account and "
              "solve any CAPTCHA. When done, return here and press Enter. <<<")
        try:
            input()
        except EOFError:
            await page.wait_for_timeout(60000)  # no stdin (e.g. piped) -> wait 60s
        await ctx.close()
    print("Profile saved. The hidden app browser will now reuse this session.")


def fetch_ai_overview_sync(query: str, **kwargs) -> dict:
    """Blocking wrapper: runs the async scraper in a dedicated thread + event
    loop. Safe to call from Streamlit (which has no usable running loop), and
    mirrors how scraper.py drives Playwright off the main thread on Windows.
    """
    import threading

    box: dict = {}

    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["result"] = loop.run_until_complete(fetch_ai_overview(query, **kwargs))
        except Exception as exc:  # noqa: BLE001 - surfaced to caller
            box["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


if __name__ == "__main__":
    import sys

    # Windows consoles default to cp1252 and choke on ₹ / — etc. in the overview.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import os

    # Reuse a persistent profile so cookies/trust build up across runs (this is
    # what keeps Google serving the overview instead of throttling you). Override
    # with CHROME_PROFILE to point at your real Chrome profile (close Chrome first).
    prof = os.getenv("CHROME_PROFILE") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".chrome-profile"
    )

    # One-time: `python google_ai_overview.py --login` to sign in and warm up.
    if sys.argv[1:2] == ["--login"]:
        asyncio.run(warmup_login(prof))
        sys.exit(0)

    q = " ".join(sys.argv[1:]) or "What is the total wheat produced by Madhya Pradesh?"
    res = asyncio.run(fetch_ai_overview(q, user_data_dir=prof, headless=False, debug=True))
    print("QUERY:", res["query"])
    if res["unavailable"]:
        print("\n[!] Google declined to show an AI Overview for this search "
              "(no overview, or this browser/IP is being throttled).")
    print("\nAI OVERVIEW TEXT:\n", res["text"] or "(empty — see ai_overview_debug.png)")
    print("\nLINKS:", res["links"])
