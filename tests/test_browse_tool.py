"""The `browse` tool: one live tab driven by click/type/scroll/back/read.
Playwright is never launched; the driven browser is a recorder."""
import asyncio

import pytest

import browser as real_browser
import server


class _Driver:
    def __init__(self):
        self.calls = []
        self.page = real_browser.PageText(
            title="Shop", url="https://shop.example/cart",
            text="Cart is empty.", char_count=14, truncated=False)

    async def drive_navigate(self, url): self.calls.append(("open", url))
    async def drive_click(self, t): self.calls.append(("click", t))
    async def drive_type(self, text, into=""): self.calls.append(("type", text, into))
    async def drive_scroll(self, d="down"): self.calls.append(("scroll", d))
    async def drive_back(self): self.calls.append(("back",))
    async def drive_read(self): return self.page


@pytest.fixture
def drv(monkeypatch):
    d = _Driver()
    monkeypatch.setattr(server, "_driven_browser", d)
    return d


def run(args):
    return asyncio.run(asyncio.wait_for(server.tool_browse(args), 5))


def test_open_click_type_scroll_back_each_reach_the_driver(drv):
    assert "Done." in run({"action": "open", "url": "https://shop.example"})
    run({"action": "click", "target": "Add to cart"})
    run({"action": "type", "text": "iphone", "target": "Search"})
    run({"action": "scroll", "direction": "up"})
    run({"action": "back"})
    assert drv.calls == [("open", "https://shop.example"),
                         ("click", "Add to cart"),
                         ("type", "iphone", "Search"),
                         ("scroll", "up"), ("back",)]


def test_result_carries_the_page_wrapped_as_untrusted(drv):
    out = run({"action": "read"})
    assert "Cart is empty." in out
    assert "shop.example/cart" in out
    assert "untrusted" in out


@pytest.mark.parametrize("url", ["file:///etc/passwd", "javascript:alert(1)", ""])
def test_open_refuses_non_web_addresses(drv, url):
    run({"action": "open", "url": url})
    assert drv.calls == []


def test_missing_arguments_and_bad_action_are_refused(drv):
    run({"action": "click"})
    run({"action": "type"})
    run({"action": "explode"})
    assert drv.calls == []


def test_browse_taints_the_turn_and_is_registered():
    assert "browse" in server.TOOL_HANDLERS
    assert "browse" in server.TAINTING_TOOLS
    assert "browse" in server.ACTING_TOOLS
