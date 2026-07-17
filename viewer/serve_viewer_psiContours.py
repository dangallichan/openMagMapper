from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import webbrowser
import os

VIEWER_DIR = Path(__file__).resolve().parent
ROOT = VIEWER_DIR.parent
PORT = 8080


def main():
    os.chdir(ROOT)
    url = f"http://127.0.0.1:{PORT}/viewer/index_psiContours.html"
    print(f"Serving {ROOT} at {url}")
    print("Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    server = ThreadingHTTPServer(("127.0.0.1", PORT), SimpleHTTPRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
