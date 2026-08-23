#!/usr/bin/env python3
"""Static file server with HTTP Range support (so browsers can seek inside the narration clips).
usage: python3 tools/serve.py [port] [directory]   — default port 8791, directory = site root"""
import os, sys, re, pathlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path) or "Range" not in self.headers:
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found"); return None
        size = os.fstat(f.fileno()).st_size
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers["Range"])
        if not m:
            f.close(); return super().send_head()
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            f.close(); self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers(); return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        f.seek(start); self._range = (start, end - start + 1)
        return f

    def copyfile(self, source, outputfile):
        rng = getattr(self, "_range", None)
        if not rng: return super().copyfile(source, outputfile)
        remaining = rng[1]
        while remaining > 0:
            chunk = source.read(min(65536, remaining))
            if not chunk: break
            outputfile.write(chunk); remaining -= len(chunk)

    def end_headers(self):
        if self.command == "GET" and "Range" not in self.headers: self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
    root = sys.argv[2] if len(sys.argv) > 2 else str(pathlib.Path(__file__).resolve().parent.parent)
    os.chdir(root)
    RangeHandler.extensions_map.update({".m4a": "audio/mp4", ".mp4": "video/mp4", ".mjs": "text/javascript", ".js": "text/javascript"})
    print(f"serving {root} on http://127.0.0.1:{port} (Range enabled)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), RangeHandler).serve_forever()
