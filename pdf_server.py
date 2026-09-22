"""Serve source PDFs on 127.0.0.1 so a browser can honor #page=N."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import quote, unquote, urlparse

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
HIGHLIGHT_DIR = ROOT / "data" / "highlights"


class PdfHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def guess_type(self, path):
        if str(path).lower().endswith(".pdf"):
            return "application/pdf"
        return super().guess_type(path)

    def translate_path(self, path):
        raw = unquote(urlparse(path).path)
        if raw.startswith("/hl/"):
            root = HIGHLIGHT_DIR.resolve()
            relative = raw[len("/hl/"):]
        else:
            root = SOURCE_DIR.resolve()
            relative = raw.lstrip("/")
        if not relative or ".." in Path(relative).parts:
            return str(root / "__missing__")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return str(root / "__missing__")
        return str(candidate)

    def do_GET(self):
        target = Path(self.translate_path(self.path))
        if not target.is_file() or target.suffix.lower() != ".pdf":
            self.send_error(404, "PDF not found")
            return
        super().do_GET()

    def do_HEAD(self):
        target = Path(self.translate_path(self.path))
        if not target.is_file() or target.suffix.lower() != ".pdf":
            self.send_error(404, "PDF not found")
            return
        super().do_HEAD()


def start_pdf_server(source_dir=None):
    HIGHLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    directory = str((source_dir or SOURCE_DIR).resolve())
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(PdfHandler, directory=directory))
    Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def page_url(port, filename, page):
    return f"http://127.0.0.1:{int(port)}/{quote(filename, safe='/')}#page={int(page)}"


def highlight_url(port, stored_name, page):
    return f"http://127.0.0.1:{int(port)}/hl/{quote(stored_name)}#page={int(page)}"
