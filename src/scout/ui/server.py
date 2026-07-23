"""A minimal local web UI that shows scout's reports live as they land on disk.

Read-only and stdlib-only on purpose: this is a viewer for the JSON/Markdown
files `scout research` and `scout investigate` already write under
`reports_dir`, not a new source of truth or a service worth a dependency.
It polls the directory tree for changes on a background thread per connected
browser tab and pushes a Server-Sent Event when something changed; the page
just refetches the whole snapshot on that signal rather than diffing, which is
simpler and cheap enough given how small these reports are.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

STATIC_DIR = Path(__file__).parent / "static"
_RUN_INDEX_PREFIX = "_run-"
_POLL_INTERVAL_S = 1.0


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _scan_day(day_dir: Path) -> dict[str, Any]:
    """One `reports/<date>/` directory: research runs plus standalone briefs.

    A research run is a `_run-*.json` index plus the per-entity md/json pairs
    it lists. A `scout investigate` brief has no run index -- it's just a
    `<slug>.json` sitting on its own -- so anything not referenced by a run
    index and shaped like a brief (has "subject") is picked up separately.
    """
    runs: list[dict[str, Any]] = []
    referenced: set[str] = set()

    for index_path in sorted(day_dir.glob(f"{_RUN_INDEX_PREFIX}*.json")):
        data = _read_json(index_path)
        if data is None:
            continue
        entries = []
        for entry in data.get("entries", []):
            full = dict(entry)
            # Run indexes written before incremental save (or by an older
            # scout version) have no "status" -- every entry in that format
            # was already a finished report.
            full.setdefault("status", "done")
            json_name = entry.get("json")
            if json_name:
                referenced.add(json_name)
                detail = _read_json(day_dir / json_name)
                if detail is not None:
                    full["detail"] = detail
            entries.append(full)
        runs.append({**data, "entries": entries, "kind": "research"})

    briefs: list[dict[str, Any]] = []
    for json_path in sorted(day_dir.glob("*.json")):
        name = json_path.name
        if name.startswith(_RUN_INDEX_PREFIX) or name in referenced:
            continue
        detail = _read_json(json_path)
        if detail is None or "subject" not in detail:
            continue  # not a shape we recognize -- skip rather than guess
        briefs.append({**detail, "json": name, "kind": "investigate"})

    return {"date": day_dir.name, "runs": runs, "briefs": briefs}


def build_snapshot(reports_dir: Path) -> dict[str, Any]:
    """Everything under `reports_dir`, newest day first."""
    if not reports_dir.exists():
        return {"reports_dir": str(reports_dir), "days": []}
    days = [
        _scan_day(day_dir)
        for day_dir in sorted((p for p in reports_dir.iterdir() if p.is_dir()), reverse=True)
    ]
    return {"reports_dir": str(reports_dir.resolve()), "days": days}


def _dir_signature(reports_dir: Path) -> tuple:
    """A cheap fingerprint of every JSON file's (path, mtime, size).

    Good enough to detect "something changed" without diffing content --
    the run index is rewritten on every completed candidate, so its mtime
    alone is enough to trigger a refresh.
    """
    if not reports_dir.exists():
        return ()
    sig = []
    for path in sorted(reports_dir.rglob("*.json")):
        try:
            stat = path.stat()
        except OSError:
            continue
        sig.append((str(path.relative_to(reports_dir)), stat.st_mtime_ns, stat.st_size))
    return tuple(sig)


def _make_handler(reports_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ScoutUI/1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass  # quiet -- this shares a terminal with `scout research` output

        def do_GET(self) -> None:  # stdlib method name, not snake_case by choice
            path = urlsplit(self.path).path
            if path == "/":
                self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif path == "/api/reports":
                self._serve_json(build_snapshot(reports_dir))
            elif path == "/api/events":
                self._serve_events()
            else:
                self.send_error(404)

        def _serve_file(self, path: Path, content_type: str) -> None:
            if not path.is_file():
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_json(self, data: Any) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last: tuple | None = None
            try:
                while True:
                    sig = _dir_signature(reports_dir)
                    if sig != last:
                        last = sig
                        self.wfile.write(b"event: update\ndata: {}\n\n")
                    else:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    time.sleep(_POLL_INTERVAL_S)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return  # the browser tab closed or navigated away

    return Handler


def serve(
    reports_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Start the server and block until interrupted (Ctrl+C).

    Binds to `port` if free, otherwise lets the OS pick one -- so a second
    `scout ui` (or anything else already on 8765) doesn't just crash.
    """
    handler = _make_handler(reports_dir)
    try:
        httpd = ThreadingHTTPServer((host, port), handler)
    except OSError:
        httpd = ThreadingHTTPServer((host, 0), handler)

    actual_port = httpd.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"scout ui  ->  {url}  (watching {reports_dir.resolve()})")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
