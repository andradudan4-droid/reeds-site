"""Render the Flask pages to plain HTML for Cloudflare.

Run from the repo root after changing any page in app.py:
    python3 tools/build_static.py
It writes everything Cloudflare serves into public/.
"""
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app as site  # noqa: E402

OUT = ROOT / "public"

PAGES = {
    "/": "index.html",
    "/services": "services.html",
    "/gallery": "gallery.html",
    "/contact": "contact.html",
    "/privacy-policy": "privacy-policy.html",
    "/terms": "terms.html",
    "/widget-frame": "widget-frame.html",
}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    client = site.app.test_client()

    for path, name in PAGES.items():
        res = client.get(path)
        assert res.status_code == 200, (path, res.status_code)
        html = res.get_data(as_text=True)
        if name == "widget-frame.html":
            # The chat logic lives in chat-frame.js; the frame's look is unchanged.
            html, n = re.subn(r"<script>.*?</script>", '<script src="/chat-frame.js?v=1"></script>', html, flags=re.S)
            assert n == 1
        (OUT / name).write_text(html)

    for path, name in (("/widget.js", "widget.js"), ("/sitemap.xml", "sitemap.xml"), ("/robots.txt", "robots.txt")):
        (OUT / name).write_text(client.get(path).get_data(as_text=True))

    # Only files committed to git, so stray local images don't get published.
    tracked = subprocess.run(["git", "ls-files", "static"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    for rel in tracked:
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dest)
    for extra in ("chat-frame.js", "_headers", "_redirects"):
        shutil.copy(ROOT / "cloudflare" / extra, OUT / extra)
    print("built", len(list(OUT.rglob("*"))), "files into public/")


if __name__ == "__main__":
    main()
