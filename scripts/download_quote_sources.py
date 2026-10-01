"""Download the public-domain translations used to verify the quotes (stdlib only)."""
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "StoicInvestorQuoteCheck/1.0 (personal portfolio project; one-off download)"}

SOURCES = {
    # Marcus Aurelius, tr. George Long (1862)
    "meditations_long_pg15877.txt": "https://www.gutenberg.org/cache/epub/15877/pg15877.txt",
    # Epictetus, tr. George Long: the complete Discourses, Encheiridion and Fragments (OCR of the Bell edition)
    "epictetus_long_complete_archive.txt": "https://archive.org/download/discoursesofepic00epicuoft/discoursesofepic00epicuoft_djvu.txt",
}


def get(url: str, tries: int = 6) -> bytes:
    """GET with a polite User-Agent; on HTTP 429 wait as asked (or 15 s) and retry."""
    import urllib.error

    for attempt in range(tries):
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == tries - 1:
                raise
            time.sleep(float(exc.headers.get("Retry-After") or 15))
    raise RuntimeError("unreachable")


def html_to_text(html: str) -> str:
    """Rendered Wikisource HTML -> plain text, one paragraph per line, footnote marks dropped."""
    from html.parser import HTMLParser

    class Strip(HTMLParser):
        def __init__(self):
            super().__init__()
            self.out, self.skip = [], 0

        def handle_starttag(self, tag, attrs):
            cls = dict(attrs).get("class") or ""
            if self.skip or tag in ("sup", "style", "script") or "reference" in cls or "mw-editsection" in cls:
                self.skip += 1
            elif tag in ("p", "div", "br", "li", "h2", "h3"):
                self.out.append("\n")

        def handle_endtag(self, tag):
            if self.skip:
                self.skip -= 1

        def handle_startendtag(self, tag, attrs):
            if tag == "br" and not self.skip:
                self.out.append("\n")

        def handle_data(self, data):
            if not self.skip:
                self.out.append(data)

    p = Strip()
    p.feed(html)
    lines = [" ".join(line.split()) for line in "".join(p.out).splitlines()]
    return "\n".join(line for line in lines if line)


def main():
    import json

    for name, url in SOURCES.items():
        path = HERE / name
        if not path.exists():
            path.write_bytes(get(url))
        print(name, path.stat().st_size)
    letters = HERE / "seneca_gummere"
    letters.mkdir(exist_ok=True)
    missing = []
    for n in range(1, 125):
        path = letters / f"letter_{n:03d}.txt"
        if path.exists() and path.stat().st_size > 2000:
            continue
        url = (
            "https://en.wikisource.org/w/api.php?action=parse&format=json&formatversion=2&prop=text"
            f"&page=Moral_letters_to_Lucilius/Letter_{n}"
        )
        try:
            html = json.loads(get(url))["parse"]["text"]
            path.write_text(html_to_text(html), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - report and carry on
            missing.append((n, str(exc)))
        time.sleep(1.2)
    print("seneca letters", len(list(letters.glob("letter_*.txt"))), "missing", missing)


if __name__ == "__main__":
    main()
