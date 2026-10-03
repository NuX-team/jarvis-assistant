#!/usr/bin/env python3
"""JARVIS — terminal client.

No browser, no HUD: a plain terminal session. Starts the same server.py
backend headlessly (no Vite, no Chrome tab) and talks to its own
`/ws/voice` WebSocket from this Python process instead — the exact protocol
the browser client already uses (echo rejection, barge-in, ack-timeouts),
just driven from a terminal.

Mic audio goes to the SAME `/ws/voice` endpoint as the browser, through a
small addition to the server (`?transcriber=gemini` — see server.py's
voice_handler): since this client has no browser SpeechRecognition to lean
on, Gemini 3.5 Transcribe runs server-side instead and feeds the same
"transcript"/"interim" messages into the existing pipeline.

Run: python3 jarvis_cli.py
Stop: Ctrl+C, or say "goodbye jarvis" / "exit".

The HUD-based browser client (`python server.py` + `cd frontend && npm run
dev`) still exists unchanged for anyone who wants it; this is a second,
independent way to reach the same server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("jarvis.cli")
for noisy in ("httpx", "httpcore", "websockets", "uvicorn", "uvicorn.access", "uvicorn.error"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

HERE = Path(__file__).parent


def _parse_env_lines(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            out.append((k.strip(), v.strip().strip('"').strip("'")))
    return out


_env_path = HERE / ".env"
if _env_path.exists():
    for _k, _v in _parse_env_lines(_env_path.read_text()):
        os.environ.setdefault(_k, _v)

USER_TITLE = os.getenv("HONORIFIC", "sir")
EXIT_PHRASES = {"goodbye jarvis", "exit", "quit", "stop listening"}
PORT = int(os.getenv("JARVIS_PORT", "8341"))   # separate from the browser server's default 8340
HOST = "127.0.0.1"


def _start_server_in_background() -> None:
    """Run server.py's FastAPI app with uvicorn, in a background thread of
    this same process — no subprocess, no second `claude` login, no Vite."""
    import uvicorn
    os.environ.setdefault("JARVIS_PORT", str(PORT))
    os.environ.setdefault("JARVIS_SCHEME", "http")
    os.environ.setdefault("JARVIS_BIND_HOST", HOST)

    import server as server_module

    config = uvicorn.Config(server_module.app, host=HOST, port=PORT, log_level="error")
    uv_server = uvicorn.Server(config)

    def _run():
        asyncio.run(uv_server.serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


async def _wait_for_server(timeout: float = 30.0) -> bool:
    import httpx
    deadline = time.monotonic() + timeout
    url = f"http://{HOST}:{PORT}/api/health"
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return False


class TerminalVoiceClient:
    """Connects to the running server's /ws/voice over a plain websocket
    and drives mic audio in, text/audio replies out — a non-browser
    implementation of exactly what frontend/src/main.ts + voice.ts do."""

    def __init__(self, ws):
        self.ws = ws
        self._printed_status = ""

    async def run(self) -> None:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def _callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

        stream = sd.RawInputStream(
            samplerate=16000, channels=1, dtype="int16",
            callback=_callback, blocksize=1600,
        )
        sender = asyncio.create_task(self._send_audio_loop(audio_queue))
        try:
            with stream:
                await self._receive_loop()
        finally:
            sender.cancel()

    async def _send_audio_loop(self, audio_queue: asyncio.Queue) -> None:
        while True:
            chunk = await audio_queue.get()
            try:
                await self.ws.send(chunk)
            except Exception:
                return

    async def _receive_loop(self) -> None:
        async for raw in self.ws:
            if isinstance(raw, (bytes, bytearray)):
                continue  # audio playback bytes are not used here; see _play below
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type")
            if kind == "status":
                state = msg.get("state")
                if state == "thinking" and self._printed_status != "thinking":
                    print("  (thinking...)")
                self._printed_status = state or ""
            elif kind == "transcript_echo":
                # Server-side Gemini transcription result, echoed back so the
                # terminal can show what it understood (see server.py addition).
                text = str(msg.get("text", ""))
                final = bool(msg.get("isFinal"))
                if final:
                    sys.stdout.write("\r" + " " * 80 + "\r")
                    print(f"You: {text}")
                else:
                    sys.stdout.write(f"\r  ...{text[-60:]}" + " " * 10)
                    sys.stdout.flush()
            elif kind == "audio":
                text = msg.get("text")
                if text:
                    print(f"JARVIS: {text}")
                data = msg.get("data")
                if data:
                    await self._play_b64_mp3(data)
                await self.ws.send(json.dumps({
                    "type": "played", "utt": msg.get("utt"), "idx": msg.get("idx")}))
            elif kind == "text":
                print(f"JARVIS: {msg.get('text', '')}")

    async def _play_b64_mp3(self, b64: str) -> None:
        import base64
        import tempfile
        audio = base64.b64decode(b64)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio)
            path = f.name
        try:
            proc = await asyncio.create_subprocess_exec(
                "afplay", path,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await proc.wait()
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass


async def main() -> None:
    print("Starting JARVIS (terminal mode)...")
    _start_server_in_background()
    if not await _wait_for_server():
        print("Server didn't come up in time — check the logs above for a crash.")
        sys.exit(1)
    print(f"JARVIS is ready, {USER_TITLE}. Listening — say \"goodbye jarvis\" to stop, or Ctrl+C.\n")

    import data_paths
    import websockets
    token = data_paths.ensure_tool_token()
    uri = f"ws://{HOST}:{PORT}/ws/voice?mode=terminal"
    try:
        async with websockets.connect(
            uri, max_size=None,
            additional_headers={"Authorization": f"Bearer {token}"},
        ) as ws:
            client = TerminalVoiceClient(ws)
            await client.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error(f"connection lost: {e}")
    print(f"\nGoodbye, {USER_TITLE}.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
