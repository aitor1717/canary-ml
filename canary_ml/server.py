from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        route  = parsed.path.rstrip("/") or "/"
        qs     = parse_qs(parsed.query)

        if route in ("", "/", "/dashboard"):
            self._serve_file(
                Path(__file__).parent / "dashboard.html",
                "text/html; charset=utf-8",
            )
        elif route == "/api/data":
            try:
                n = int(qs.get("n", ["100"])[0])
                if n <= 0:
                    n = 100
            except (ValueError, TypeError):
                self.send_response(400)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Bad request: 'n' must be a positive integer")
                return
            from canary_ml.storage import MonitorLog
            entries = MonitorLog(self.server.log_path).read_last(n)
            self._json(entries)
        elif route == "/api/reference":
            ref = Path(self.server.log_path) / "reference.json"
            self._json(json.loads(ref.read_text()) if ref.exists() else None)
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def start(
    log_path: str | Path,
    port: int = 8501,
    *,
    host: str = "127.0.0.1",
    block: bool = False,
) -> HTTPServer:
    """Start the canary dashboard server.

    host="127.0.0.1" (default): localhost-only, matching the printed URL.
    Pass host="0.0.0.0" to make the dashboard reachable from other machines
    on the network — only do this if you understand that monitor.jsonl
    (and the unauthenticated /api/data, /api/reference endpoints) may
    contain raw feature data.

    block=False (default): runs in a daemon thread — use inside ModelMonitor.
    block=True: blocks the calling thread — use when running as __main__.
    """
    server = HTTPServer((host, port), _Handler)
    server.log_path = str(log_path)
    if block:
        server.serve_forever()
    else:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    import sys
    _log  = sys.argv[1] if len(sys.argv) > 1 else "./canary_logs"
    _port = int(sys.argv[2]) if len(sys.argv) > 2 else 8501
    try:
        from rich.console import Console
        Console().print(f"[bold yellow]canary[/bold yellow] dashboard → http://localhost:{_port}")
    except ImportError:
        print(f"canary dashboard → http://localhost:{_port}")
    start(_log, _port, block=True)
