"""Check relative page and anchor links in a completed documentation build."""

from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote

root = Path(__file__).resolve().parents[2] / "site"


class Page(HTMLParser):
    def __init__(self, p):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(p.read_text())

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "a" and "href" in a:
            self.links.append(a["href"])


pages = {p.resolve(): Page(p) for p in root.rglob("*.html")}
if not pages:
    raise SystemExit("Build documentation first: site contains no HTML pages.")

bad = set()
for p, page in pages.items():
    for href in page.links:
        u = urlsplit(href)
        if u.scheme or u.netloc or u.path.startswith("/"):
            continue
        target = (p.parent / unquote(u.path)).resolve() if u.path else p
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            bad.add((str(p.relative_to(root)), href, "missing page"))
        elif (
            u.fragment
            and target in pages
            and unquote(u.fragment) not in pages[target].ids
        ):
            bad.add((str(p.relative_to(root)), href, "missing anchor"))
for item in sorted(bad):
    print(*item, sep=" | ")
print("Broken local links:", len(bad))

if bad:
    raise SystemExit(1)
