"""Batch classification API for the Censer.9 Hub bundle (stdlib only, no FastAPI).

    python serve.py --port 8000                   # serves the local HF bundle dir (run from the bundle root)
    CENSER_DIR=/path/to/bundle python scripts/serve.py

Endpoints:
    GET  /health          -> {"status": "ok", "model": ...}
    POST /classify        -> {"texts": [...]}  => {"results": [{label, unsafe, score, categories, chunks}]}
    POST /classify-one    -> {"text": "..."}   => single result

Features: batching, long-input chunking (chunk_tokens/stride), token-bucket
rate limit (env RATE_LIMIT_PER_MIN, default 600), max body size guard.

    python serve.py --dir bundle --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY = 2 * 1024 * 1024  # 2 MB


class RateLimiter:
    def __init__(self, per_min: float) -> None:
        self.per_min = per_min
        self.tokens = per_min
        self.updated = time.monotonic()
        self.last_deny = 0.0

    def allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        self.tokens = min(self.per_min, self.tokens + (now - self.updated) * self.per_min / 60.0)
        self.updated = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


def make_handler(pipe, limiter: RateLimiter, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # quiet
            pass

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok", "model": model_name, "device": str(pipe.device)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not limiter.allow():
                self._json(429, {"error": "rate limit exceeded"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._json(413, {"error": "body too large"})
                return
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._json(400, {"error": "invalid json"})
                return
            if self.path == "/classify-one":
                text = str(payload.get("text", ""))
                self._json(200, pipe.predict(text))
            elif self.path == "/classify":
                texts = payload.get("texts")
                if not isinstance(texts, list) or not texts or len(texts) > 256:
                    self._json(400, {"error": "texts must be a list of 1..256 strings"})
                    return
                self._json(200, {"results": pipe.predict_batch([str(t) for t in texts])})
            else:
                self._json(404, {"error": "not found"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.environ.get("CENSER_DIR", "bundle"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from censer import CenserPipeline

    pipe = CenserPipeline(args.dir, device=args.device)
    limiter = RateLimiter(float(os.environ.get("RATE_LIMIT_PER_MIN", "600")))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(pipe, limiter, args.dir))
    print(f"censer serve: model={args.dir} device={pipe.device} on http://{args.host}:{args.port}")
    print("  POST /classify {\"texts\": [...]}   POST /classify-one {\"text\": ...}   GET /health")
    server.serve_forever()


if __name__ == "__main__":
    main()
