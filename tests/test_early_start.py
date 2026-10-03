"""Early start: the brain begins on the first words, not after the pause."""
import asyncio

import pytest

import gemini_transcribe as g


def make(monkeypatch, ms=20):
    monkeypatch.setenv("JARVIS_EARLY_START_MS", str(ms))
    got = {"early": [], "final": [], "interim": []}

    async def i(t): got["interim"].append(t)
    async def f(t): got["final"].append(t)
    async def e(t): got["early"].append(t)
    async def err(m): pass
    return g.GeminiTranscriber(i, f, err, on_early=e), got


async def feed(tr, text):
    tr._last_text = g.merge_chunk(tr._last_text, text)
    tr._arm_early()


def test_merge_handles_cumulative_and_fragment_styles():
    assert g.merge_chunk("open goo", "open google") == "open google"
    assert g.merge_chunk("open", " google") == "open google"
    assert g.merge_chunk("", "hi") == "hi"


def test_fires_early_once_the_text_holds_still(monkeypatch):
    async def go():
        tr, got = make(monkeypatch)
        await feed(tr, "open google and search")
        await asyncio.sleep(0.1)
        assert got["early"] == ["open google and search"]
    asyncio.run(go())


def test_too_few_words_waits(monkeypatch):
    async def go():
        tr, got = make(monkeypatch)
        await feed(tr, "open google")
        await asyncio.sleep(0.1)
        assert got["early"] == []
    asyncio.run(go())


def test_still_talking_resets_the_clock(monkeypatch):
    async def go():
        tr, got = make(monkeypatch, ms=80)
        await feed(tr, "open google and")
        await asyncio.sleep(0.04)
        await feed(tr, " search for iphone")
        await asyncio.sleep(0.2)
        assert got["early"] == ["open google and search for iphone"]
    asyncio.run(go())


def test_final_after_early_is_not_run_twice(monkeypatch):
    async def go():
        tr, got = make(monkeypatch)
        await feed(tr, "open google and search")
        await asyncio.sleep(0.1)
        await tr._finish_turn("open google and search")
        assert got["final"] == []
    asyncio.run(go())


def test_words_after_early_become_an_addendum(monkeypatch):
    async def go():
        tr, got = make(monkeypatch)
        await feed(tr, "open google and search")
        await asyncio.sleep(0.1)
        await tr._finish_turn("open google and search for iphone prices")
        assert got["final"] == [g.ADDENDUM_PREFIX + "for iphone prices"]
    asyncio.run(go())


def test_no_early_means_old_behaviour(monkeypatch):
    async def go():
        tr, got = make(monkeypatch, ms=0)
        await feed(tr, "open google and search")
        await asyncio.sleep(0.05)
        assert got["early"] == []
        await tr._finish_turn("open google and search")
        assert got["final"] == ["open google and search"]
    asyncio.run(go())
