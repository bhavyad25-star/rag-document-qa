
from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Find the saved project files
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "embedded_chunks.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(file_path):
    """Load chunk text and metadata from a JSONL file."""
    chunks = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def search_chunks(question, model, embeddings, chunks, top_k=3):
    """Return the top-k chunks most similar to a question."""

    # 2. Convert the question into an embedding
    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    # 3. Compare the question with every chunk embedding
    scores = embeddings @ question_embedding

    # 4. Get the indices of the highest scores first
    best_indices = np.argsort(scores)[::-1][:top_k]

    # 5. Display the matching chunks and their metadata
    for rank, index in enumerate(best_indices, start=1):
        chunk = chunks[index]
        score = scores[index]

        print(f"\n--- Result {rank} ---")
        print(f"Similarity score: {score:.4f}")
        print(f"Source: {chunk['source']}")
        print(f"Page: {chunk['page']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Text:\n{chunk['text']}")


def main():
    # 6. Check that the saved files exist
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {EMBEDDINGS_FILE}")

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {CHUNKS_FILE}")

    # 7. Load the saved vectors and chunk records
    embeddings = np.load(EMBEDDINGS_FILE)
    chunks = load_chunks(CHUNKS_FILE)

    if len(embeddings) != len(chunks):
        raise ValueError("The number of embeddings and chunks does not match.")

    # 8. Load the same embedding model used earlier
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loaded {len(chunks)} chunks.")
    print("Type 'exit' to stop.")

    # 9. Keep accepting questions
    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() == "exit":
            print("Search finished.")
            break

        if not question:
            print("Please enter a question.")
            continue

        search_chunks(question, model, embeddings, chunks, top_k=3)


if __name__ == "__main__":
    main()
