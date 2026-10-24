"""Local HTTP server for the observer page: static files, an SSE event stream, and read-only memory files."""

import json
import logging
import queue
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pokemon_agent.memory.store import MemoryStore
from pokemon_agent.observer.hub import ObserverHub

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
KEEPALIVE_SECONDS = 15


def _make_handler(hub: ObserverHub, memory: MemoryStore):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            logger.debug("observer: " + format, *args)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/events":
                self._stream_events()
            elif url.path == "/memory":
                self._memory_file(parse_qs(url.query).get("path", [""])[0])
            else:
                self._static(url.path)

        def _send(self, status: HTTPStatus, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, path: str):
            relative = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() not in target.parents or not target.is_file():
                self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8")
                return
            content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
            self._send(HTTPStatus.OK, target.read_bytes(), content_type)

        def _memory_file(self, path: str):
            result = memory.execute("view", {"path": path, "forceReadLargeFiles": True})
            self._send(HTTPStatus.OK, result.encode(), "text/plain; charset=utf-8")

        def _stream_events(self):
            subscriber, snapshot = hub.subscribe()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self._write_event(snapshot)
                while not subscriber.dropped.is_set():
                    try:
                        event = subscriber.queue.get(timeout=KEEPALIVE_SECONDS)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self._write_event(event)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            finally:
                hub.unsubscribe(subscriber)

        def _write_event(self, event: dict):
            payload = json.dumps(event, default=str, separators=(",", ":"))
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()

    return Handler


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Browsers drop event-stream connections whenever a tab closes or reloads; that is not an error.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class ObserverServer:
    """Serves the observer page on a background thread."""

    def __init__(self, hub: ObserverHub, memory: MemoryStore, host: str = "127.0.0.1", port: int = 8765):
        self.hub = hub
        self.httpd = _QuietServer((host, port), _make_handler(hub, memory))
        self.url = f"http://{host}:{self.httpd.server_port}/"
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="observer-http", daemon=True)

    def start(self) -> "ObserverServer":
        self._thread.start()
        logger.info(f"Observer page: {self.url}")
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.hub.close()
