"""
JARVIS Browser — Playwright-based web browsing capabilities.

Provides search, page visits, screenshots, and multi-step research.
Runs headless Chromium with realistic user agent to avoid blocking.
"""

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger("jarvis.browser")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
TIMEOUT_MS = 30_000


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageContent:
    title: str
    url: str
    text_content: str
    word_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResearchResult:
    topic: str
    sources: list[str]
    summary: str
    key_findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Browser Manager
# ---------------------------------------------------------------------------

class JarvisBrowser:
    """Playwright-based web browsing for JARVIS."""

    def __init__(self):
        self._pw = None
        self._browser = None
        self._context = None
        self._driven_page = None

    async def _ensure_browser(self):
        """Launch browser if not running."""
        if self._browser and self._context:
            return

        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        # Launch VISIBLE browser so user can watch JARVIS browse
        self._browser = await self._pw.chromium.launch(headless=False)
        self._context = await self._browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 900},
        )
        log.info("Browser launched (visible Chromium)")

    async def _new_page(self):
        """Create a new page in the browser context."""
        await self._ensure_browser()
        return await self._context.new_page()

    # -- Search ----------------------------------------------------------------

    async def search(self, query: str) -> list[SearchResult]:
        """Search DuckDuckGo and return top results."""
        page = await self._new_page()
        results = []

        try:
            await page.goto(
                f"https://html.duckduckgo.com/html/?q={query}",
                timeout=TIMEOUT_MS,
                wait_until="domcontentloaded",
            )

            # Extract search results from DDG HTML version
            raw = await page.evaluate("""
                () => {
                    const items = document.querySelectorAll('.result');
                    return Array.from(items).slice(0, 5).map(item => ({
                        title: (item.querySelector('.result__title a') || item.querySelector('.result__a'))?.textContent?.trim() || '',
                        url: (item.querySelector('.result__title a') || item.querySelector('.result__a'))?.href || '',
                        snippet: item.querySelector('.result__snippet')?.textContent?.trim() || ''
                    }));
                }
            """)

            for r in raw:
                if r.get("title") and r.get("url"):
                    results.append(SearchResult(
                        title=r["title"],
                        url=r["url"],
                        snippet=r.get("snippet", ""),
                    ))

            log.info(f"Search '{query}' returned {len(results)} results")
            # Let user see the search results for a moment
            await asyncio.sleep(2)
        except Exception as e:
            log.warning(f"Search failed for '{query}': {e}")
        finally:
            # Don't close the page — keep it visible
            pass

        return results

    # -- Visit URL -------------------------------------------------------------

    async def visit(self, url: str) -> PageContent:
        """Visit a URL and extract main text content."""
        page = await self._new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)

            data = await page.evaluate("""
                () => {
                    const title = document.title || '';

                    // Try to get main content area first
                    const main = document.querySelector('main')
                        || document.querySelector('article')
                        || document.querySelector('[role="main"]')
                        || document.body;

                    // Remove noise elements
                    const clone = main.cloneNode(true);
                    for (const el of clone.querySelectorAll(
                        'script, style, nav, header, footer, aside, .sidebar, .menu, .ad, .advertisement, iframe'
                    )) {
                        el.remove();
                    }

                    const text = clone.innerText || clone.textContent || '';
                    // Trim to reasonable size
                    const trimmed = text.substring(0, 5000).trim();
                    return {
                        title: title,
                        text: trimmed,
                    };
                }
            """)

            text = data.get("text", "")
            return PageContent(
                title=data.get("title", ""),
                url=url,
                text_content=text,
                word_count=len(text.split()),
            )

            # Let user see the page for a moment
            await asyncio.sleep(3)
        except Exception as e:
            log.warning(f"Visit failed for '{url}': {e}")
            return PageContent(
                title="Error",
                url=url,
                text_content=f"Failed to load page: {e}",
                word_count=0,
            )
        # Don't close — keep pages visible

    # -- Screenshot ------------------------------------------------------------

    async def screenshot(self, url: str, path: str = None) -> str:
        """Take screenshot of a page. Returns file path to PNG."""
        page = await self._new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
            await page.wait_for_timeout(1000)  # let rendering settle

            if not path:
                tmp = tempfile.mktemp(suffix=".png", prefix="jarvis_screenshot_")
                path = tmp

            await page.screenshot(path=path, full_page=True)
            log.info(f"Screenshot saved: {path}")
            return path

        except Exception as e:
            log.warning(f"Screenshot failed for '{url}': {e}")
            return ""
        finally:
            await page.close()

    # -- Research (multi-step) -------------------------------------------------

    async def research(self, topic: str) -> ResearchResult:
        """Multi-step research: search -> visit top results -> compile findings."""
        results = await self.search(topic)
        sources = []
        contents = []

        for r in results[:3]:
            try:
                page_content = await self.visit(r.url)
                sources.append(r.url)
                contents.append(
                    f"## {r.title}\nURL: {r.url}\n\n{page_content.text_content[:1500]}"
                )
            except Exception:
                continue

        summary = "\n\n---\n\n".join(contents) if contents else "No results found."

        return ResearchResult(
            topic=topic,
            sources=sources,
            summary=summary,
            key_findings=[r.title for r in results[:3]],
        )

    # -- Lifecycle -------------------------------------------------------------

    async def close(self):
        """Shut down the browser."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._pw:
                await self._pw.stop()
            log.info("Browser closed")
        except Exception as e:
            log.warning(f"Browser close error: {e}")
        finally:
            self._pw = None
            self._browser = None
            self._context = None
            self._driven_page = None

    # -- Interactive driving (click/type/navigate on one live page) -----------
    #
    # The user asked for JARVIS to "control" the browser, not just open a URL
    # and read it: click things, type into fields, go to a new address, all
    # on the SAME visible tab across several voice turns, so the user watches
    # one continuous session rather than a fresh window per action. This is
    # a single page (`self._driven_page`), separate from `_new_page()`'s
    # throwaway tabs used by search/visit/screenshot, created lazily on the
    # first drive_* call and reused until `close()` or `drive_navigate` is
    # told to open a NEW tab.

    async def _driven(self):
        if self._driven_page is None or self._driven_page.is_closed():
            self._driven_page = await self._new_page()
        return self._driven_page

    async def drive_navigate(self, url: str) -> str:
        """Go to `url` on the driven tab. Returns the page title."""
        page = await self._driven()
        await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        await page.wait_for_timeout(150)
        return await page.title()

    async def drive_back(self) -> str:
        page = await self._driven()
        await page.go_back(wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        return await page.title()

    async def drive_click(self, text: str) -> str:
        """Click the first visible element whose text/label/placeholder/
        aria-label matches `text` (case-insensitive, partial match) —
        covers buttons, links, and labelled inputs without the caller
        needing to know CSS selectors."""
        page = await self._driven()
        locator = page.get_by_text(text, exact=False).first
        try:
            await locator.wait_for(state="visible", timeout=3000)
        except Exception:
            # Fall back to role-based matching (buttons/links without
            # exact visible text, e.g. an icon button with an aria-label).
            locator = page.get_by_role("button", name=text, exact=False).first
            try:
                await locator.wait_for(state="visible", timeout=3000)
            except Exception:
                raise PageError(f"couldn't find anything on the page matching {text!r}")
        await locator.click(timeout=5000)
        await page.wait_for_timeout(100)
        return await page.title()

    async def drive_type(self, text: str, into: str = "") -> str:
        """Type `text` into the field identified by `into` (its label,
        placeholder, or aria-label), or into whatever is currently focused
        if `into` is empty."""
        page = await self._driven()
        if into:
            locator = page.get_by_label(into, exact=False).first
            try:
                await locator.wait_for(state="visible", timeout=3000)
            except Exception:
                locator = page.get_by_placeholder(into, exact=False).first
                await locator.wait_for(state="visible", timeout=3000)
            await locator.fill(text, timeout=5000)
        else:
            await page.keyboard.type(text)
        return await page.title()

    async def drive_scroll(self, direction: str = "down") -> None:
        page = await self._driven()
        amount = 600 if direction == "down" else -600
        await page.mouse.wheel(0, amount)

    async def drive_read(self) -> "PageText":
        """What's on the driven tab right now — same shape as read_page,
        but against the live interactive session rather than a throwaway
        headless tab."""
        page = await self._driven()
        data = await page.evaluate(_EXTRACT_JS, PAGE_TEXT_CHARS)
        text = str(data.get("text") or "")
        full = int(data.get("full") or len(text))
        return PageText(title=str(data.get("title") or ""), url=page.url,
                        text=text, char_count=full, truncated=full > len(text))


# ---------------------------------------------------------------------------
# Looking at a page for JARVIS himself
# ---------------------------------------------------------------------------
#
# The user, twice: "when I tell you to open a website it'd be great if we
# could look at things together ... you can understand everything that I'm
# actually seeing visually and/or you get a really quick data back of the
# content that's on the page so you can read it very quick."
#
# HEADLESS, deliberately, and unlike `JarvisBrowser` above.
#
# `JarvisBrowser` launches `headless=False` on purpose: it is the "watch me
# browse" path, where a visible window IS the feature, and it keeps its pages
# open afterwards so the user can look at them. These two functions are the
# opposite job. They exist so the BRAIN can read or see a page in the middle
# of a spoken turn — for itself, not for the user. Showing the user a page is
# already `open_in_browser`, which uses their own real browser with their own
# logins and extensions. A second, visible, cookie-less Chromium popping up
# and stealing focus every time JARVIS glanced at a URL would be a bug, not a
# feature: this project treats taking the user's focus as something only an
# ACTING tool may do, and reading is not acting.
#
# One throwaway browser per call, always closed. No shared state with
# `JarvisBrowser`, so a glance can never leave a window behind, and can never
# inherit a half-dead context from a research run an hour ago.

# Every one of these must finish WELL inside `jarvis_mcp.TIMEOUT_SEC` (20s).
# A handler that outlives that tells the brain the server is unreachable
# while the work carries on regardless — the exact lie documented at the top
# of jarvis_mcp.py. Launch (~0.5s) + navigate (<=12s) + capture (~1s) leaves
# real headroom, and the caller puts its own hard deadline on top of this.
LOOK_TIMEOUT_MS = 12_000

# 1280x800 is a laptop window: wide enough that a site renders its desktop
# layout rather than its mobile one, small enough that the PNG stays a few
# hundred KB and the image costs the brain on the order of a thousand tokens.
LOOK_VIEWPORT = {"width": 1280, "height": 800}

# What the page evaluator may hand back. This is NOT the brain's budget —
# that is the caller's, and it is far smaller. This only stops a pathological
# page from moving megabytes over the loopback hop before anyone trims it.
PAGE_TEXT_CHARS = 20_000

# A viewport-sized PNG bigger than this is not a screenshot of a web page, it
# is something that will not fit through the tool channel. Refuse it out loud
# rather than sending something the CLI will choke on.
MAX_SHOT_BYTES = 4_000_000


class PageError(Exception):
    """A page JARVIS could not read or see. The message is speakable."""


@dataclass
class PageText:
    title: str
    url: str
    text: str
    char_count: int          # the extracted length BEFORE the caller's budget
    truncated: bool          # PAGE_TEXT_CHARS clipped it


@dataclass
class PageShot:
    title: str
    url: str
    png: bytes


# The same idea as `visit()`, with two differences that matter inside a small
# budget:
#
# 1. the noise is removed from the LIVE document, not from a detached clone.
#    `innerText` is defined in terms of RENDERED text, and a clone that is not
#    in the document has no layout — so `clone.innerText` silently degrades to
#    something textContent-shaped and runs every block together. Measured: a
#    heading and the paragraph under it came back as "ARC REACTOR STATUSOutput
#    is nominal", one unreadable word. Mutating the page is free here: the tab
#    is headless, private to this call, and destroyed a few lines later.
# 2. the bound is passed IN rather than hard-coded in the middle of a JS
#    string, so it cannot drift from the Python constant that documents it.
_EXTRACT_JS = """
(limit) => {
    const main = document.querySelector('main')
        || document.querySelector('article')
        || document.querySelector('[role="main"]')
        || document.body;
    for (const el of main.querySelectorAll(
        'script, style, noscript, nav, header, footer, aside, .sidebar, .menu, .ad, .advertisement, iframe'
    )) {
        el.remove();
    }
    const text = (main.innerText || main.textContent || '')
        .replace(/[ \\t]+/g, ' ')
        .replace(/\\n{3,}/g, '\\n\\n')
        .trim();
    return {title: document.title || '', full: text.length,
            text: text.substring(0, limit)};
}
"""


class _Headless:
    """One throwaway headless Chromium, guaranteed closed.

    `async with _Headless() as page:` — every exit path, exception included,
    shuts the browser and stops Playwright. A leaked Chromium here would
    outlive JARVIS himself.
    """

    def __init__(self):
        self._pw = None
        self._browser = None

    async def __aenter__(self):
        from playwright.async_api import async_playwright
        self._pw = await async_playwright().start()
        try:
            self._browser = await self._pw.chromium.launch(headless=True)
            context = await self._browser.new_context(
                user_agent=USER_AGENT, viewport=dict(LOOK_VIEWPORT))
            page = await context.new_page()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        page.set_default_timeout(LOOK_TIMEOUT_MS)
        return page

    async def __aexit__(self, *exc):
        for close in (getattr(self._browser, "close", None),
                      getattr(self._pw, "stop", None)):
            if close is None:
                continue
            try:
                await close()
            except Exception as e:                  # never mask the real error
                log.warning(f"headless teardown: {e}")
        self._pw = None
        self._browser = None
        return False


async def read_page(url: str) -> PageText:
    """The readable text of one page. Raises PageError if it cannot be had."""
    try:
        async with _Headless() as page:
            await page.goto(url, wait_until="domcontentloaded",
                            timeout=LOOK_TIMEOUT_MS)
            data = await page.evaluate(_EXTRACT_JS, PAGE_TEXT_CHARS)
            landed = page.url or url
    except PageError:
        raise
    except Exception as e:
        log.warning(f"read_page failed for {url}: {e}")
        raise PageError("that page wouldn't load") from e

    if not isinstance(data, dict):
        raise PageError("that page had no readable text on it")
    text = str(data.get("text") or "")
    full = int(data.get("full") or len(text))
    if not text.strip():
        raise PageError("that page had no readable text on it")
    return PageText(title=str(data.get("title") or ""), url=str(landed),
                    text=text, char_count=full, truncated=full > len(text))


async def capture_page(url: str) -> PageShot:
    """A viewport-sized PNG of one page. Raises PageError if it cannot be had.

    Viewport, NOT `full_page`: a full-page capture of a long article is a tall
    thin strip that costs the brain a great many tokens and shows it what a
    person would only see by scrolling. What the user means by "look at this
    with me" is the screenful in front of him.
    """
    try:
        async with _Headless() as page:
            await page.goto(url, wait_until="domcontentloaded",
                            timeout=LOOK_TIMEOUT_MS)
            # Let web fonts, images and any late layout settle. A screenshot
            # of a half-painted page is worse than no screenshot at all.
            await page.wait_for_timeout(700)
            title = await page.title()
            png = await page.screenshot(type="png", full_page=False)
            landed = page.url or url
    except PageError:
        raise
    except Exception as e:
        log.warning(f"capture_page failed for {url}: {e}")
        raise PageError("I couldn't get a picture of that page") from e

    if not png:
        raise PageError("I couldn't get a picture of that page")
    if len(png) > MAX_SHOT_BYTES:
        raise PageError("that page's picture came out far too large to send")
    return PageShot(title=str(title or ""), url=str(landed), png=bytes(png))
