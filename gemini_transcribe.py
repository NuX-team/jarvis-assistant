"""Gemini 3.5 Transcribe (Live API) — streaming speech-to-text.

Replaces the browser's own SpeechRecognition (Chrome -> Google's older Web
Speech engine) with Google's dedicated transcription model over a persistent
WebSocket session, sent directly from the server: the browser just forwards
raw PCM16 audio and we talk to Gemini.

Why server-side rather than browser-side: `google-genai`'s Live client is a
Python asyncio API and this project already runs one event loop per voice
connection in `server.py`; doing it server-side also keeps the Gemini API key
off the browser entirely.

Protocol: `gemini-3.5-transcribe-live`, response_modalities=["TEXT"],
input_audio_transcription configured with `language_codes` (empty = auto
language detection, which the model also supports). JARVIS_SPEECH_LANG in
.env sets the expected language code list; defaults to Uzbek since that is
what the browser's SpeechRecognition could not do at all (it was hard-coded
to en-US and transcribed Uzbek speech AS English garbage).

Audio must arrive as raw 16-bit PCM, 16 kHz, mono, little-endian — the
frontend's AudioWorklet (see `frontend/src/gemini-audio.ts`) resamples the
mic to this format before sending.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Awaitable, Callable, Optional

log = logging.getLogger("jarvis.gemini_transcribe")

MODEL = "gemini-3.5-transcribe-live"

# Empty list = Gemini auto-detects the spoken language on its own, which the
# model supports natively. A fixed list is more accurate for one known
# speaker/language and reconnects faster, so default to it; "auto" in the
# .env value switches to auto-detection instead.
_DEFAULT_LANG = "uz-UZ"


def _configured_langs() -> list[str]:
    raw = os.getenv("JARVIS_SPEECH_LANG", _DEFAULT_LANG).strip()
    if not raw or raw.lower() == "auto":
        return []
    return [raw]


def _early_ms() -> int:
    """How long the transcript must hold still before the brain starts on it.
    0 turns early start off (wait for the end-of-speech signal, as before)."""
    try:
        return max(0, int(os.getenv("JARVIS_EARLY_START_MS", "600")))
    except ValueError:
        return 600


EARLY_MIN_WORDS = 3
# What the brain is told about words that arrive after it already started.
ADDENDUM_PREFIX = "(addition to my previous request) "


def merge_chunk(current: str, chunk: str) -> str:
    """Live transcription can arrive as the full text so far OR as a new
    fragment. If the chunk extends what we have, it replaces it; otherwise it
    is appended — so neither style loses the start of the sentence."""
    if not current:
        return chunk
    if chunk.startswith(current) or chunk.strip() == current.strip():
        return chunk
    return current + chunk


def api_key_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


class GeminiTranscriber:
    """One Gemini Live transcription session, bound to one voice WebSocket.

    `start()` opens the session and begins a background receive loop that
    calls `on_interim`/`on_final` as transcripts arrive. `send_audio(chunk)`
    pushes PCM16/16kHz audio in; `stop()` closes the session. Any failure to
    start or a dropped connection mid-session is reported through `on_error`
    rather than raised, so the caller can fall back to the browser's own
    recognizer instead of losing voice input entirely.
    """

    def __init__(
        self,
        on_interim: Callable[[str], Awaitable[None]],
        on_final: Callable[[str], Awaitable[None]],
        on_error: Callable[[str], Awaitable[None]],
        on_early: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self._on_early = on_early
        self._early_task: Optional[asyncio.Task] = None
        self._committed = ""          # text the brain has already started on
        self._on_interim = on_interim
        self._on_final = on_final
        self._on_error = on_error
        self._session = None
        self._session_cm = None
        self._receive_task: Optional[asyncio.Task] = None
        self._client = None
        self._last_text = ""

    async def start(self) -> bool:
        """Open the Live API session. Returns False (and calls on_error)
        if the API key is missing or the connection fails, so the caller
        can fall back rather than silently losing voice input."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            await self._on_error("no Gemini API key configured")
            return False
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            await self._on_error("google-genai package not installed")
            return False

        try:
            self._client = genai.Client(api_key=api_key)
            config = types.LiveConnectConfig(
                response_modalities=["TEXT"],
                input_audio_transcription=types.AudioTranscriptionConfig(
                    language_codes=_configured_langs(),
                ),
            )
            self._session_cm = self._client.aio.live.connect(model=MODEL, config=config)
            self._session = await self._session_cm.__aenter__()
        except Exception as e:
            log.error(f"Gemini transcribe session failed to open: {e}")
            await self._on_error(f"couldn't reach Gemini: {e}")
            self._session = None
            self._session_cm = None
            return False

        self._receive_task = asyncio.create_task(self._receive_loop())
        log.info(f"Gemini transcribe session open (langs={_configured_langs() or 'auto'})")
        return True

    async def send_audio(self, pcm16_bytes: bytes) -> None:
        if self._session is None:
            return
        from google.genai import types
        try:
            await self._session.send_realtime_input(
                audio=types.Blob(data=pcm16_bytes, mime_type="audio/pcm;rate=16000")
            )
        except Exception as e:
            log.warning(f"Gemini transcribe send failed: {e}")
            await self._on_error("lost the Gemini connection")

    async def _receive_loop(self) -> None:
        """Stream transcripts back. Gemini reports incremental text on
        `input_transcription` within a turn; `turn_complete` marks a final
        utterance boundary (voice-activity-detected end of speech)."""
        try:
            async for response in self._session.receive():
                content = getattr(response, "server_content", None)
                if content is None:
                    continue
                transcription = getattr(content, "input_transcription", None)
                if transcription is not None and transcription.text:
                    self._last_text = merge_chunk(self._last_text, transcription.text)
                    await self._on_interim(self._last_text)
                    self._arm_early()
                if getattr(content, "turn_complete", False) and self._last_text.strip():
                    final = self._last_text
                    self._last_text = ""
                    await self._finish_turn(final)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"Gemini transcribe receive loop ended: {e}")
            await self._on_error("the Gemini connection dropped")

    # -- Early start ----------------------------------------------------------
    #
    # The end-of-speech signal arrives after the user has stopped AND a pause
    # has elapsed, so waiting for it costs the whole tail of every command.
    # Instead: once the transcript has held still for `_early_ms()` and has
    # enough words to be a task, hand it to the brain while the user may still
    # be finishing. Words that come after are sent as an addendum.

    def _arm_early(self) -> None:
        delay = _early_ms()
        if self._on_early is None or delay <= 0 or self._committed:
            return
        if self._early_task is not None:
            self._early_task.cancel()
        self._early_task = asyncio.create_task(self._early_after(delay / 1000.0))

    async def _early_after(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        text = self._last_text.strip()
        if self._committed or len(text.split()) < EARLY_MIN_WORDS:
            return
        self._committed = text
        try:
            await self._on_early(text)
        except Exception as e:
            log.warning(f"early start failed: {e}")

    async def _finish_turn(self, final: str) -> None:
        if self._early_task is not None:
            self._early_task.cancel()
            self._early_task = None
        committed, self._committed = self._committed, ""
        if not committed:
            await self._on_final(final)
            return
        final = final.strip()
        rest = final[len(committed):].strip() if final.startswith(committed) else ""
        if len(rest.split()) >= 2:
            await self._on_final(ADDENDUM_PREFIX + rest)

    async def stop(self) -> None:
        if self._early_task is not None:
            self._early_task.cancel()
            self._early_task = None
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._receive_task = None
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception as e:
                log.warning(f"Gemini transcribe session close: {e}")
            self._session_cm = None
        self._session = None
