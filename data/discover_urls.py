"""One-off crawler: starting from a few uwaterloo.ca hub pages, find same-section
subpages worth including in the corpus. Prints one URL per line. Not part of the
main pipeline -- run manually to (re)generate seed_urls.txt.
"""
import sys
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

SEEDS = [
    "https://uwaterloo.ca/co-operative-education/",
    "https://uwaterloo.ca/housing/",
    "https://uwaterloo.ca/housing/residences",
    "https://uwaterloo.ca/orientation/",
    "https://uwaterloo.ca/orientation/international-students",
    "https://uwaterloo.ca/student-success/",
    "https://uwaterloo.ca/registrar/",
    "https://uwaterloo.ca/international-students",
    "https://uwaterloo.ca/campus-wellness/",
]

# Only keep links that stay within these path prefixes -- keeps the corpus
# focused on first-year / co-op / student-life topics instead of the whole site.
ALLOWED_PREFIXES = (
    "/co-operative-education",
    "/housing",
    "/orientation",
    "/student-success",
    "/registrar",
    "/international-students",
    "/campus-wellness",
)

HEADERS = {"User-Agent": "Mozilla/5.0 (student resume project; contact via github)"}


def same_section(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc not in ("uwaterloo.ca", "www.uwaterloo.ca"):
        return False
    return any(parsed.path.startswith(p) for p in ALLOWED_PREFIXES)


def main():
    seen = set()
    ordered = []
    for seed in SEEDS:
        try:
            resp = requests.get(seed, headers=HEADERS, timeout=10)
        except requests.RequestException as exc:
            print(f"ERROR fetching {seed}: {exc}", file=sys.stderr)
            continue
        if resp.status_code != 200:
            print(f"SKIP {seed} ({resp.status_code})", file=sys.stderr)
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.select("a[href]"):
            href = urljoin(seed, a["href"]).split("#")[0]
            if "?" in href:
                continue
            if same_section(href) and href not in seen:
                seen.add(href)
                ordered.append(href)
        time.sleep(0.3)

    for url in SEEDS + ordered:
        print(url)


if __name__ == "__main__":
    main()
