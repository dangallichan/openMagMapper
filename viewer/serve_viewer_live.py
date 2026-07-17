from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import csv
import json
import os
import threading
import webbrowser

VIEWER_DIR = Path(__file__).resolve().parent
ROOT = VIEWER_DIR.parent
PORT = 8081
LIVE_FRAME_PATH = VIEWER_DIR / "live_frame.jpg"


class FrozenCsvTailCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._path = None
        self._rows = []
        self._count = 0
        self._offset = 0
        self._size = 0
        self._header_idxs = None

    def _reset_for_path(self, path: Path):
        self._path = path
        self._rows = []
        self._count = 0
        self._offset = 0
        self._size = 0
        self._header_idxs = None

    def _load_header(self, file_obj) -> bool:
        header_raw = file_obj.readline()
        if not header_raw:
            self._offset = file_obj.tell()
            return False

        header_line = header_raw.decode("utf-8", errors="ignore").strip("\r\n")
        try:
            cols = next(csv.reader([header_line]))
        except Exception:
            self._offset = file_obj.tell()
            return False

        required = {
            "sensor_table_x_m": None,
            "sensor_table_y_m": None,
            "sensor_table_z_m": None,
            "mag_table_x_uT": None,
            "mag_table_y_uT": None,
            "mag_table_z_uT": None,
        }
        for i, name in enumerate(cols):
            if name in required:
                required[name] = i

        if any(v is None for v in required.values()):
            self._header_idxs = None
            self._offset = file_obj.tell()
            return False

        self._header_idxs = required
        self._offset = file_obj.tell()
        return True

    def _parse_data_lines(self, file_obj):
        if self._header_idxs is None:
            return

        ix = self._header_idxs["sensor_table_x_m"]
        iy = self._header_idxs["sensor_table_y_m"]
        iz = self._header_idxs["sensor_table_z_m"]
        ivx = self._header_idxs["mag_table_x_uT"]
        ivy = self._header_idxs["mag_table_y_uT"]
        ivz = self._header_idxs["mag_table_z_uT"]

        for raw_line in file_obj:
            line = raw_line.decode("utf-8", errors="ignore").strip("\r\n")
            if not line:
                continue

            try:
                cols = next(csv.reader([line]))
                row = {
                    "origin": [
                        float(cols[ix]),
                        float(cols[iy]),
                        float(cols[iz]),
                    ],
                    "vec": [
                        float(cols[ivx]),
                        float(cols[ivy]),
                        float(cols[ivz]),
                    ],
                }
                self._rows.append(row)
                self._count += 1
            except Exception:
                continue

        self._offset = file_obj.tell()

    def get_tail(self, path: Path, since: int):
        with self._lock:
            if self._path != path:
                self._reset_for_path(path)

            try:
                current_size = path.stat().st_size
            except Exception:
                return 0, []

            if current_size < self._size:
                self._reset_for_path(path)

            if self._offset == 0:
                with path.open("rb") as f:
                    if self._load_header(f):
                        self._parse_data_lines(f)
                    self._size = current_size
            elif current_size > self._size:
                with path.open("rb") as f:
                    f.seek(self._offset)
                    self._parse_data_lines(f)
                self._size = current_size

            safe_since = max(0, min(int(since), self._count))
            tail = self._rows[safe_since:]
            return self._count, tail


FROZEN_CACHE = FrozenCsvTailCache()


def find_latest_frozen_csv() -> Path | None:
    candidates = []
    for rel in [Path("data") / "experimentData", Path("dataKeep")]:
        folder = ROOT / rel
        if not folder.exists():
            continue
        candidates.extend(folder.glob("Exp_cam_frozenVectors_*.csv"))

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


class LiveViewerHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/frozen_tail":
            query = parse_qs(parsed.query)
            try:
                since = int(query.get("since", ["0"])[0])
            except Exception:
                since = 0
            since = max(0, since)

            csv_path = find_latest_frozen_csv()
            if csv_path is None:
                self._send_json(
                    {
                        "source": "",
                        "count": 0,
                        "rows": [],
                        "message": "No frozen vectors CSV found yet.",
                    }
                )
                return

            count, tail = FROZEN_CACHE.get_tail(csv_path, since)
            self._send_json(
                {
                    "source": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                    "count": count,
                    "rows": tail,
                    "mtime": csv_path.stat().st_mtime,
                }
            )
            return

        if parsed.path == "/api/live_frame.jpg":
            if not LIVE_FRAME_PATH.exists():
                self.send_error(404, "Live frame not available yet")
                return

            data = LIVE_FRAME_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()
            self.wfile.write(data)
            return

        return super().do_GET()


def main():
    os.chdir(ROOT)
    url = f"http://127.0.0.1:{PORT}/viewer/index_live.html"
    print(f"Serving {ROOT} at {url}")
    print("Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    server = ThreadingHTTPServer(("127.0.0.1", PORT), LiveViewerHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
