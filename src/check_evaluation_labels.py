
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHUNKS_FILE = DATA_DIR / "embedded_chunks.jsonl"
QUESTIONS_FILE = DATA_DIR / "evaluation_questions.jsonl"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    chunks = read_jsonl(CHUNKS_FILE)
    questions = read_jsonl(QUESTIONS_FILE)

    available_pairs = {
        (chunk.get("source"), chunk.get("page"))
        for chunk in chunks
    }

    missing = []

    for number, item in enumerate(questions, start=1):
        expected_source = item["relevant_source"]
        expected_pages = item["relevant_pages"]

        found = any(
            (expected_source, page) in available_pairs
            for page in expected_pages
        )

        if not found:
            missing.append((number, item))

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Loaded evaluation questions: {len(questions)}")
    print(f"Questions with available source/page labels: "
          f"{len(questions) - len(missing)}/{len(questions)}")

    if missing:
        print("\nMISSING LABELS:")
        for number, item in missing:
            print(f"\nQuestion {number}: {item['question']}")
            print(f"Expected source: {item['relevant_source']}")
            print(f"Expected pages: {item['relevant_pages']}")

        raise SystemExit(
            "\nLabel check failed. Inspect the missing labels."
        )

    print("\nLabel check PASSED!")


if __name__ == "__main__":
    main()

