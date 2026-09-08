"""Fetch each URL in seed_urls.txt, strip boilerplate (nav/header/footer/scripts),
and save the readable page text to data/raw/<slug>.txt with a small metadata header.

Usage: python data/ingest.py
"""
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HERE = Path(__file__).parent
SEED_FILE = HERE / "seed_urls.txt"
RAW_DIR = HERE / "raw"

HEADERS = {"User-Agent": "Mozilla/5.0 (student resume project; contact via github)"}

# Tags/selectors that are boilerplate on uwaterloo.ca pages, not article content.
STRIP_SELECTORS = [
    "script", "style", "nav", "header", "footer", "noscript",
    "[role=navigation]", ".breadcrumb", ".uw-nav", "#block-uwaterloo-branding-navigation",
    "#block-uwaterloo-branding-footer", ".skip-link", "form",
]


def load_urls() -> list[str]:
    urls = []
    for line in SEED_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def slugify(url: str) -> str:
    path = urlparse(url).path.strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-") or "index"
    return slug


def extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    main = soup.find("main") or soup.find(id="main-content") or soup.body or soup
    for sel in STRIP_SELECTORS:
        for tag in main.select(sel):
            tag.decompose()

    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if text:
            lines.append(text)

    body = "\n".join(lines)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return title, body


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urls = load_urls()
    ok, skipped = 0, 0

    for i, url in enumerate(urls, 1):
        slug = slugify(url)
        out_path = RAW_DIR / f"{slug}.txt"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
        except requests.RequestException as exc:
            print(f"[{i}/{len(urls)}] ERROR {url}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        if resp.status_code != 200:
            print(f"[{i}/{len(urls)}] SKIP {url} ({resp.status_code})", file=sys.stderr)
            skipped += 1
            continue

        title, body = extract_text(resp.text)
        if len(body) < 200:
            print(f"[{i}/{len(urls)}] SKIP {url} (too little content: {len(body)} chars)", file=sys.stderr)
            skipped += 1
            continue

        out_path.write_text(f"URL: {url}\nTITLE: {title}\n\n{body}\n")
        print(f"[{i}/{len(urls)}] OK {url} -> {out_path.name} ({len(body)} chars)")
        ok += 1
        time.sleep(0.2)

    print(f"\nDone. {ok} pages saved, {skipped} skipped.")


if __name__ == "__main__":
    main()
