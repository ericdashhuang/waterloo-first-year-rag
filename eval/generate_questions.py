"""Generate an evaluation question for each source page, using an LLM.

Reads every file in data/raw/, asks an LLM to write one realistic question a
first-year student would ask that this specific page answers, and saves the
result as eval/questions.json -- a small ground-truth set for measuring
retrieval accuracy (see eval/run_eval.py).

Needs an API key: set ANTHROPIC_API_KEY or GROQ_API_KEY (e.g. in a .env
file). Groq's free tier is enough for this -- ~73 short requests, one per
source page.

Usage: python eval/generate_questions.py
"""
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))
from chunking import parse_raw_file  # noqa: E402

HERE = Path(__file__).parent
RAW_DIR = HERE.parent / "data" / "raw"
OUT_PATH = HERE / "questions.json"

PROMPT = """Below is the text of a real page from the University of Waterloo's \
website, aimed at first-year students. Write ONE realistic question a \
first-year student would actually type into a search box, that this page \
specifically answers. Keep it under 20 words, and don't reference "this page" \
or "the text above". Return only the question itself, no quotes, no preamble.

Page title: {title}

Page text:
{body}"""


def ask_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def ask_groq(prompt: str) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def ask(title: str, body: str) -> str:
    prompt = PROMPT.format(title=title, body=body[:3000])
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ask_anthropic(prompt)
    if os.environ.get("GROQ_API_KEY"):
        return ask_groq(prompt)
    raise SystemExit("Set ANTHROPIC_API_KEY or GROQ_API_KEY (e.g. in a .env file) first.")


def main():
    load_dotenv()
    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        raise SystemExit(f"No files found in {RAW_DIR} -- run data/ingest.py first.")

    questions = []
    for i, path in enumerate(raw_files, 1):
        url, title, body = parse_raw_file(path.read_text())
        question = ask(title, body)
        questions.append({"question": question, "source_url": url, "source_title": title})
        print(f"  [{i}/{len(raw_files)}] {title}: {question}")

    OUT_PATH.write_text(json.dumps(questions, indent=2))
    print(f"\nWrote {len(questions)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
