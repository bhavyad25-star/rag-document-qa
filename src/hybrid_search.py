
from pathlib import Path
import json
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
)
CHUNKS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "embedded_chunks.jsonl"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
RRF_K = 60


def load_chunks(file_path):
    chunks = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def get_rankings(question, model, embeddings, chunks, bm25):
    # 1. Rank all chunks using dense retrieval.
    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    dense_scores = embeddings @ query_embedding

    dense_order = np.argsort(dense_scores)[::-1]

    # Convert each chunk's position into a 1-based rank.
    dense_ranks = np.empty(len(chunks), dtype=int)

    for rank, index in enumerate(dense_order, start=1):
        dense_ranks[index] = rank

    # 2. Rank all chunks using BM25.
    query_words = tokenize(question)
    bm25_scores = bm25.get_scores(query_words)

    bm25_order = sorted(
        range(len(chunks)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    bm25_ranks = np.empty(len(chunks), dtype=int)

    for rank, index in enumerate(bm25_order, start=1):
        bm25_ranks[index] = rank

    # 3. Combine both rankings using Reciprocal Rank Fusion.
    rrf_scores = (
        1.0 / (RRF_K + dense_ranks)
        + 1.0 / (RRF_K + bm25_ranks)
    )

    hybrid_order = np.argsort(rrf_scores)[::-1]

    return [
        {
            "index": int(index),
            "rrf_score": float(rrf_scores[index]),
            "dense_rank": int(dense_ranks[index]),
            "bm25_rank": int(bm25_ranks[index]),
            "dense_score": float(dense_scores[index]),
            "bm25_score": float(bm25_scores[index]),
        }
        for index in hybrid_order
    ]


def main():
    for path in [EMBEDDINGS_FILE, CHUNKS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    embeddings = np.load(EMBEDDINGS_FILE)
    chunks = load_chunks(CHUNKS_FILE)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "The number of embeddings and chunks does not match."
        )

    # Build BM25's index using exactly the same chunk order.
    tokenized_chunks = [
        tokenize(chunk["text"]) for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_chunks)

    # Load the same embedding model used by dense retrieval.
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loaded {len(chunks)} chunks.")
    print("Hybrid search is ready.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() == "exit":
            print("Search finished.")
            break

        if not question:
            print("Please enter a question.")
            continue

        if not tokenize(question):
            print("No searchable words found.")
            continue

        results = get_rankings(
            question, model, embeddings, chunks, bm25
        )

        for result_number, result in enumerate(
            results[:TOP_K], start=1
        ):
            chunk = chunks[result["index"]]

            print(f"\n--- Hybrid Result {result_number} ---")
            print(f"RRF score: {result['rrf_score']:.6f}")
            print(f"Dense rank: {result['dense_rank']}")
            print(f"BM25 rank: {result['bm25_rank']}")
            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page']}")
            print(f"Chunk ID: {chunk.get('chunk_id', 'N/A')}")
            print(f"Text:\n{chunk['text']}")


if __name__ == "__main__":
    main()

from pathlib import Path
import json
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
)
CHUNKS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "embedded_chunks.jsonl"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
RRF_K = 60


def load_chunks(file_path):
    chunks = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def get_rankings(question, model, embeddings, chunks, bm25):
    # 1. Rank all chunks using dense retrieval.
    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    dense_scores = embeddings @ query_embedding

    dense_order = np.argsort(dense_scores)[::-1]

    # Convert each chunk's position into a 1-based rank.
    dense_ranks = np.empty(len(chunks), dtype=int)

    for rank, index in enumerate(dense_order, start=1):
        dense_ranks[index] = rank

    # 2. Rank all chunks using BM25.
    query_words = tokenize(question)
    bm25_scores = bm25.get_scores(query_words)

    bm25_order = sorted(
        range(len(chunks)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    bm25_ranks = np.empty(len(chunks), dtype=int)

    for rank, index in enumerate(bm25_order, start=1):
        bm25_ranks[index] = rank

    # 3. Combine both rankings using Reciprocal Rank Fusion.
    rrf_scores = (
        1.0 / (RRF_K + dense_ranks)
        + 1.0 / (RRF_K + bm25_ranks)
    )

    hybrid_order = np.argsort(rrf_scores)[::-1]

    return [
        {
            "index": int(index),
            "rrf_score": float(rrf_scores[index]),
            "dense_rank": int(dense_ranks[index]),
            "bm25_rank": int(bm25_ranks[index]),
            "dense_score": float(dense_scores[index]),
            "bm25_score": float(bm25_scores[index]),
        }
        for index in hybrid_order
    ]


def main():
    for path in [EMBEDDINGS_FILE, CHUNKS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    embeddings = np.load(EMBEDDINGS_FILE)
    chunks = load_chunks(CHUNKS_FILE)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "The number of embeddings and chunks does not match."
        )

    # Build BM25's index using exactly the same chunk order.
    tokenized_chunks = [
        tokenize(chunk["text"]) for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_chunks)

    # Load the same embedding model used by dense retrieval.
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loaded {len(chunks)} chunks.")
    print("Hybrid search is ready.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() == "exit":
            print("Search finished.")
            break

        if not question:
            print("Please enter a question.")
            continue

        if not tokenize(question):
            print("No searchable words found.")
            continue

        results = get_rankings(
            question, model, embeddings, chunks, bm25
        )

        for result_number, result in enumerate(
            results[:TOP_K], start=1
        ):
            chunk = chunks[result["index"]]

            print(f"\n--- Hybrid Result {result_number} ---")
            print(f"RRF score: {result['rrf_score']:.6f}")
            print(f"Dense rank: {result['dense_rank']}")
            print(f"BM25 rank: {result['bm25_rank']}")
            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page']}")
            print(f"Chunk ID: {chunk.get('chunk_id', 'N/A')}")
            print(f"Text:\n{chunk['text']}")


if __name__ == "__main__":
    main()
