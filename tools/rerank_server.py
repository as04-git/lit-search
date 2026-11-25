"""Local cross-encoder service that orders candidates by relevance to a hunt's question.

Runs outside lit-search's environment (torch stays out of it), e.g.:

    HF_HUB_OFFLINE=1 ~/coursework/.venv/bin/python -I tools/rerank_server.py --port 8797

Then ``lit-search config set LIT_SEARCH_RERANK_URL http://127.0.0.1:8797``.  lit-search
asks it to order each move's new candidates and falls back to its own order when the
service is down, slow or refuses.

Protocol: ``POST /rerank`` with ``{"query": str, "documents": [str, ...]}`` returns
``{"scores": [float, ...], "model": str}``; ``GET /health`` returns ``{"ok": true, ...}``.
Evidence for the default model: in a first-exposure test over 29 recorded hunts
(2026-10-16), it put target papers in an agent's first ten for 81% of exposures,
against 43% for the hunt's own order.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_DOCUMENTS = 400
MAX_CHARS = 2000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--device", default=None, help="cuda, cpu (default: cuda if available)")
    args = parser.parse_args()

    import torch
    from sentence_transformers import CrossEncoder

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"model_kwargs": {"dtype": "bfloat16"}} if device == "cuda" else {}
    model = CrossEncoder(args.model, device=device, **kwargs)
    model.max_length = 512
    lock = threading.Lock()
    print(f"rerank server: {args.model} on {device}, port {args.port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "model": args.model, "device": device})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/rerank":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length))
                query = str(request["query"])
                documents = [str(d)[:MAX_CHARS] for d in request["documents"]]
            except (ValueError, KeyError, TypeError):
                self._send(400, {"error": "expected {query, documents}"})
                return
            if len(documents) > MAX_DOCUMENTS:
                self._send(413, {"error": f"at most {MAX_DOCUMENTS} documents"})
                return
            started = time.monotonic()
            with lock:
                scores = model.predict(
                    [(query, d) for d in documents], batch_size=16, show_progress_bar=False
                )
            self._send(
                200,
                {
                    "scores": [float(s) for s in scores],
                    "model": args.model,
                    "seconds": round(time.monotonic() - started, 3),
                },
            )

        def log_message(self, format: str, *values: object) -> None:
            return

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
