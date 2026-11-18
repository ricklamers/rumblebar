"""Local HTTP transcription server for Reson8/Rumble-2-Flash.

Binds to 127.0.0.1 only. The model loads in a background thread so /health
answers immediately while weights load and the GPU warms up.

Endpoints
  GET  /health      -> {"status": "loading"|"ready"|"error", ...}
  POST /transcribe  -> {"text", "language", "duration", "elapsed"}
                       (requires an "X-Rumble-Client: 1" header)

/transcribe body is either raw little-endian float32 mono samples at 16 kHz
(Content-Type: application/octet-stream) or a PCM WAV file (audio/wav, any
sample rate / channel count; int16, int32 or float32).

Query parameters (all optional):
  language   English name to force ("Dutch"); omit or "auto" to detect
  digits     1/0  numbers as digits (default 1)
  filler     1/0  keep "um"/"uh" (default 0)
  verbatim   1/0  keep repeated words (default 1)
  context    free text appended to the system prompt to bias vocabulary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

SAMPLE_RATE = 16_000
CLIENT_HEADER = "X-Rumble-Client"
# Audio up to this length is transcribed in a single pass (80 s verified to
# work well). Longer audio is split at quiet points near CHUNK_TARGET_S.
SINGLE_PASS_MAX_S = 240
CHUNK_TARGET_S = 180
CHUNK_SEARCH_S = 20

log = logging.getLogger("rumble")


class Transcriber:
    def __init__(self, model_id: str, device: str | None):
        self.model_id = model_id
        self.requested_device = device
        self.status = "loading"
        self.error: str | None = None
        self.device = None
        self.load_seconds: float | None = None
        self._lock = threading.Lock()
        self.processor = None
        self.model = None

    # -- loading ---------------------------------------------------------
    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            t0 = time.time()
            device = self.requested_device
            if device is None:
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.bfloat16 if device != "cpu" else torch.float32
            log.info("loading %s on %s (%s)", self.model_id, device, dtype)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModelForMultimodalLM.from_pretrained(self.model_id, dtype=dtype)
            self.model = model.to(device).eval()
            self.device = device
            # Warm up kernels so the first real dictation is fast.
            self._run(np.zeros(SAMPLE_RATE, dtype=np.float32), "English", "", 8)
            self.load_seconds = round(time.time() - t0, 2)
            self.status = "ready"
            log.info("ready in %.2fs", self.load_seconds)
        except Exception as exc:  # surface to /health instead of dying silently
            log.exception("model load failed")
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"

    # -- inference -------------------------------------------------------
    def _run(self, audio: np.ndarray, language: str | None, system: str, max_new: int):
        import torch

        kwargs = {"prompt": system}
        if language:
            kwargs["language"] = language
        inputs = self.processor.apply_transcription_request(audio=audio, **kwargs)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        generated = out[:, inputs["input_ids"].shape[-1]:]
        parsed = self.processor.decode(generated, return_format="parsed")[0]
        return parsed.get("transcription", "").strip(), parsed.get("language") or language

    def transcribe(self, audio: np.ndarray, *, language: str | None, digits: bool,
                   filler: bool, verbatim: bool, context: str) -> dict:
        if self.status != "ready":
            raise RuntimeError(f"model not ready ({self.status})")
        # Control tokens go first in the system prompt; the chat template uses
        # a system prompt that already contains them as-is.
        system = (("<digits>" if digits else "<nodigits>")
                  + ("<filler>" if filler else "<nofiller>")
                  + ("<verbatim>" if verbatim else "<noverbatim>")
                  + (context or ""))
        duration = len(audio) / SAMPLE_RATE
        t0 = time.time()
        texts, detected = [], None
        with self._lock:
            for chunk in split_long_audio(audio):
                secs = len(chunk) / SAMPLE_RATE
                # ~3.7 tokens/s observed for English speech; leave ample headroom.
                max_new = int(max(64, secs * 12))
                text, lang = self._run(chunk, language, system, max_new)
                detected = detected or lang
                if text:
                    texts.append(text)
        return {
            "text": " ".join(texts),
            "language": detected,
            "duration": round(duration, 3),
            "elapsed": round(time.time() - t0, 3),
        }


def split_long_audio(audio: np.ndarray) -> list[np.ndarray]:
    """Return audio unchanged if short; otherwise cut at the quietest 100 ms
    window within ±CHUNK_SEARCH_S of each CHUNK_TARGET_S boundary."""
    if len(audio) <= SINGLE_PASS_MAX_S * SAMPLE_RATE:
        return [audio]
    win = SAMPLE_RATE // 10
    chunks, start = [], 0
    while len(audio) - start > SINGLE_PASS_MAX_S * SAMPLE_RATE:
        target = start + CHUNK_TARGET_S * SAMPLE_RATE
        lo = target - CHUNK_SEARCH_S * SAMPLE_RATE
        hi = target + CHUNK_SEARCH_S * SAMPLE_RATE
        seg = audio[lo:hi]
        n = len(seg) // win
        energy = (seg[: n * win].reshape(n, win) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * win + win // 2
        chunks.append(audio[start:cut])
        start = cut
    chunks.append(audio[start:])
    return chunks


# -- audio decoding ------------------------------------------------------
def decode_wav(data: bytes) -> np.ndarray:
    """Minimal RIFF/WAVE reader supporting PCM int16/int32 and float32."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, channels, rate, bits)
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("WAV missing fmt or data chunk")
    tag, channels, rate, bits = fmt
    if tag == 3 and bits == 32:
        samples = np.frombuffer(pcm, dtype="<f4").astype(np.float32)
    elif tag == 1 and bits == 16:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 32:
        samples = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV encoding (format {tag}, {bits}-bit)")
    if channels > 1:
        samples = samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1)
    return resample(samples, rate)


def resample(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE or len(samples) == 0:
        return samples
    n_out = int(round(len(samples) * SAMPLE_RATE / rate))
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


# -- HTTP ----------------------------------------------------------------
def flag(q: dict, name: str, default: bool) -> bool:
    v = q.get(name, [None])[0]
    if v is None or v == "":
        return default
    return v.lower() in ("1", "true", "yes", "on")


def make_handler(tx: Transcriber):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RumbleSTT/0.1"

        def log_message(self, fmt, *args):
            log.info("%s - %s", self.address_string(), fmt % args)

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path == "/health":
                self._json(200, {
                    "status": tx.status, "error": tx.error, "model": tx.model_id,
                    "device": tx.device, "load_seconds": tx.load_seconds,
                })
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/transcribe":
                return self._json(404, {"error": "not found"})
            # A custom header forces a CORS preflight (which we never grant),
            # so web pages in a browser cannot POST audio to this server.
            if self.headers.get(CLIENT_HEADER) is None:
                return self._json(403, {"error": f"missing {CLIENT_HEADER} header"})
            if tx.status != "ready":
                return self._json(503, {"error": f"model {tx.status}", "detail": tx.error})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = self.rfile.read(length)
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
                if ctype in ("audio/wav", "audio/x-wav", "audio/wave") or data[:4] == b"RIFF":
                    audio = decode_wav(data)
                else:
                    audio = np.frombuffer(data[: len(data) // 4 * 4], dtype="<f4").astype(np.float32)
            except Exception as exc:
                return self._json(400, {"error": f"bad audio: {exc}"})
            if len(audio) < SAMPLE_RATE // 4:
                return self._json(200, {"text": "", "language": None,
                                        "duration": len(audio) / SAMPLE_RATE, "elapsed": 0})
            q = parse_qs(url.query)
            language = (q.get("language", [""])[0] or "").strip()
            if language.lower() in ("", "auto"):
                language = None
            try:
                result = tx.transcribe(
                    audio, language=language,
                    digits=flag(q, "digits", True), filler=flag(q, "filler", False),
                    verbatim=flag(q, "verbatim", True), context=q.get("context", [""])[0],
                )
            except Exception as exc:
                log.exception("transcription failed")
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            log.info("transcribed %.1fs in %.2fs (%s)", result["duration"],
                     result["elapsed"], result["language"])
            self._json(200, result)

    return Handler


def watch_parent(pid: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log.info("parent %d exited; shutting down", pid)
            os._exit(0)
        except PermissionError:
            pass
        time.sleep(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="Reson8/Rumble-2-Flash")
    ap.add_argument("--device", default=None, help="mps or cpu (default: mps if available)")
    ap.add_argument("--parent-pid", type=int, default=None,
                    help="exit when this process goes away (set by the menu bar app)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.parent_pid:
        threading.Thread(target=watch_parent, args=(args.parent_pid,), daemon=True).start()
    tx = Transcriber(args.model, args.device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(tx))
    threading.Thread(target=tx.load, daemon=True).start()
    log.info("listening on http://127.0.0.1:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
