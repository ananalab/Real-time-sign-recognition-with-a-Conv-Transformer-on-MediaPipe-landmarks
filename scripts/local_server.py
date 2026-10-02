"""Static file server on a free local port, used by the headless-browser scripts."""

from __future__ import annotations

import functools
import http.server
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def serve(directory: str | Path) -> Iterator[int]:
    """Serve ``directory`` over HTTP in a background thread; yields the port."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
