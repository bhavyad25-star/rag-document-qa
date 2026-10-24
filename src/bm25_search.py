
from pathlib import Path
import json
import re
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "embedded_chunks.jsonl"


def load_chunks(file_path):
    chunks = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def tokenize(text):
    # Convert text into lowercase words.
    # For example: "Dense Retrieval Works!"
    # becomes ["dense", "retrieval", "works"]
    return re.findall(r"\b\w+\b", text.lower())


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {CHUNKS_FILE}")

    chunks = load_chunks(CHUNKS_FILE)

    # BM25 needs a list of words for each chunk.
    tokenized_chunks = [
        tokenize(chunk["text"]) for chunk in chunks
    ]

    # Build the BM25 search index.
    bm25 = BM25Okapi(tokenized_chunks)

    print(f"Loaded {len(chunks)} chunks.")
    print("BM25 keyword search is ready.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() == "exit":
            print("Search finished.")
            break

        if not question:
            print("Please enter a question.")
            continue

        query_words = tokenize(question)

        if not query_words:
            print("No searchable words found.")
            continue

        scores = bm25.get_scores(query_words)

        # Sort chunks from highest to lowest BM25 score.
        best_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:3]

        for rank, index in enumerate(best_indices, start=1):
            chunk = chunks[index]

            print(f"\n--- Result {rank} ---")
            print(f"BM25 score: {scores[index]:.4f}")
            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page']}")
            print(f"Chunk ID: {chunk.get('chunk_id', 'N/A')}")
            print(f"Text:\n{chunk['text']}")


if __name__ == "__main__":
    main()
