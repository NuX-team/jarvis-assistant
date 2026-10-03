"""Free text-to-speech fallback (edge-tts) for the terminal client.

Fish Audio's TTS API needs a paid balance (the free web tier's credits don't
carry over to API access) — this is a free alternative with no API key and
no balance to run out, so the terminal client has a voice without owing
anyone money. Falls back to macOS `say` if edge-tts has no network.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("jarvis.free_tts")

# A neural voice list: pick one that handles Uzbek/Russian/English reasonably.
# uz-UZ has no edge-tts voice as of this writing, so Uzbek text is spoken
# through the Russian voice (closest phonetic neighbor available) rather than
# silently failing; this is a known rough edge; Fish Audio remains the
# recommended path once funded. en-US is the default for English/mixed text.
DEFAULT_VOICE = os.getenv("JARVIS_FREE_TTS_VOICE", "en-US-GuyNeural")


async def synthesize_free(text: str, voice: str = DEFAULT_VOICE) -> Optional[bytes]:
    """MP3 bytes from edge-tts, or None if it failed (caller falls back to `say`)."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        import edge_tts
    except ImportError:
        log.warning("edge-tts not installed; falling back to macOS say")
        return None
    try:
        communicate = edge_tts.Communicate(text, voice)
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        return bytes(buf) if buf else None
    except Exception as e:
        log.warning(f"edge-tts failed: {e}")
        return None


def speak_with_say(text: str, voice: str = "Daniel") -> bool:
    """Last-resort offline fallback: macOS's built-in `say`, blocking."""
    text = (text or "").strip()
    if not text:
        return False
    try:
        subprocess.run(["say", "-v", voice, text], check=True, timeout=60)
        return True
    except Exception as e:
        log.error(f"macOS say failed: {e}")
        return False


async def speak(text: str, voice: str = DEFAULT_VOICE) -> None:
    """Synthesize and play `text` out loud, blocking until playback finishes.
    Tries edge-tts first (better quality), falls back to `say` (always works,
    offline, no dependency)."""
    audio = await synthesize_free(text, voice)
    if audio:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio)
            path = f.name
        try:
            proc = await asyncio.create_subprocess_exec(
                "afplay", path,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass
        return
    await asyncio.to_thread(speak_with_say, text)
