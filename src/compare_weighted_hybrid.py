
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
QUESTIONS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "evaluation_questions.jsonl"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_VALUES = [3, 5, 10]
RRF_K = 60
DENSE_WEIGHT = 1.25
BM25_WEIGHT = 1.0

def load_jsonl(file_path):
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def dense_ranking(question, model, embeddings):
    query_vector = model.encode(
        question,
        normalize_embeddings=True
    )

    scores = embeddings @ query_vector
    return np.argsort(scores)[::-1].tolist()


def bm25_ranking(question, bm25, number_of_chunks):
    query_words = tokenize(question)

    scores = bm25.get_scores(query_words)

    return sorted(
        range(number_of_chunks),
        key=lambda i: scores[i],
        reverse=True
    )


def hybrid_ranking(dense_order, bm25_order, number_of_chunks):
    # Store the 1-based rank of each chunk in both lists.
    dense_ranks = np.empty(number_of_chunks, dtype=int)
    bm25_ranks = np.empty(number_of_chunks, dtype=int)

    for rank, index in enumerate(dense_order, start=1):
        dense_ranks[index] = rank

    for rank, index in enumerate(bm25_order, start=1):
        bm25_ranks[index] = rank

    # Combine rankings with Reciprocal Rank Fusion.
    rrf_scores = (
    DENSE_WEIGHT / (RRF_K + dense_ranks)
    + BM25_WEIGHT / (RRF_K + bm25_ranks)
)

    return np.argsort(rrf_scores)[::-1].tolist()


def is_hit(ranking, chunks, expected, k):
    relevant_source = expected["relevant_source"]
    relevant_pages = expected["relevant_pages"]

    for index in ranking[:k]:
        chunk = chunks[index]

        if (
            chunk["source"] == relevant_source
            and chunk["page"] in relevant_pages
        ):
            return True

    return False


def main():
    required_files = [
        EMBEDDINGS_FILE,
        CHUNKS_FILE,
        QUESTIONS_FILE
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    embeddings = np.load(EMBEDDINGS_FILE)
    chunks = load_jsonl(CHUNKS_FILE)
    questions = load_jsonl(QUESTIONS_FILE)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "The number of embeddings and chunks does not match."
        )

    if len(questions) == 0:
        raise ValueError("No evaluation questions were found.")

    # Build BM25 from the same chunks used by dense retrieval.
    tokenized_chunks = [
        tokenize(chunk["text"]) for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_chunks)

    # Load the embedding model once for the entire evaluation.
    model = SentenceTransformer(MODEL_NAME)

    methods = ["Dense", "BM25", "Hybrid"]

    # Store each method's successful retrieval count at each k.
    hits = {
        method: {k: 0 for k in K_VALUES}
        for method in methods
    }

    print(f"Chunks: {len(chunks)}")
    print(f"Evaluation questions: {len(questions)}")

    for number, item in enumerate(questions, start=1):
        question = item["question"]

        dense_order = dense_ranking(
            question, model, embeddings
        )

        bm25_order = bm25_ranking(
            question, bm25, len(chunks)
        )

        hybrid_order = hybrid_ranking(
            dense_order, bm25_order, len(chunks)
        )

        if "retrieval-augmented generation, and why" in question.lower():
            expected = item
            print("\n--- RAG QUESTION RANK DEBUG ---")
            print("Expected source:", expected["relevant_source"])
            print("Expected pages:", expected["relevant_pages"])

            for rank, index in enumerate(hybrid_order, start=1):
                chunk = chunks[index]

                if (
                    chunk["source"] == expected["relevant_source"]
                    and chunk["page"] in expected["relevant_pages"]
                ):
                    print("First matching hybrid rank:", rank)
                    print("Matching chunk ID:", chunk.get("chunk_id", "N/A"))
                    break
            else:
                print("No matching chunk found in hybrid ranking.")

        rankings = {
            "Dense": dense_order,
            "BM25": bm25_order,
            "Hybrid": hybrid_order
        }

        print(f"\nQuestion {number}: {question}")

        for method, ranking in rankings.items():
            outcomes = []

            for k in K_VALUES:
                hit = is_hit(ranking, chunks, item, k)
                hits[method][k] += int(hit)
                outcomes.append(f"Success@{k}={hit}")

            print(f"  {method}: " + ", ".join(outcomes))

    # Print final results as percentages.
    print("\n========== FINAL COMPARISON ==========")
    print(f"Questions tested: {len(questions)}")
    print("Success means a correct source AND page was retrieved.")

    header = f"{'Method':<12}"
    for k in K_VALUES:
        header += f"{'Success@' + str(k):>12}"
    print("\n" + header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<12}"

        for k in K_VALUES:
            percentage = (
                hits[method][k] / len(questions) * 100
            )
            row += f"{percentage:>11.2f}%"

        print(row)

    print("\nSuccessful questions at each cutoff:")

    for method in methods:
        counts = ", ".join(
            f"@{k}: {hits[method][k]}/{len(questions)}"
            for k in K_VALUES
        )
        print(f"{method}: {counts}")


if __name__ == "__main__":
    main()
