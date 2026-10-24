
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
TOP_K = 5
RRF_K = 60


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
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
    order = np.argsort(scores)[::-1]

    return [
        (int(index), float(scores[index]))
        for index in order
    ]


def bm25_ranking(question, bm25, number_of_chunks):
    scores = bm25.get_scores(tokenize(question))

    order = sorted(
        range(number_of_chunks),
        key=lambda i: scores[i],
        reverse=True
    )

    return [
        (index, float(scores[index]))
        for index in order
    ]


def hybrid_ranking(dense_results, bm25_results, number_of_chunks):
    dense_ranks = np.empty(number_of_chunks, dtype=int)
    bm25_ranks = np.empty(number_of_chunks, dtype=int)

    for rank, (index, _) in enumerate(dense_results, start=1):
        dense_ranks[index] = rank

    for rank, (index, _) in enumerate(bm25_results, start=1):
        bm25_ranks[index] = rank

    rrf_scores = (
        1.0 / (RRF_K + dense_ranks)
        + 1.0 / (RRF_K + bm25_ranks)
    )

    order = np.argsort(rrf_scores)[::-1]

    return [
        (int(index), float(rrf_scores[index]))
        for index in order
    ]


def matches_expected(chunk, question):
    return (
        chunk["source"] == question["relevant_source"]
        and chunk["page"] in question["relevant_pages"]
    )


def print_results(method, ranking, chunks, question):
    print(f"\n--- {method}: Top {TOP_K} ---")

    for rank, (index, score) in enumerate(
        ranking[:TOP_K], start=1
    ):
        chunk = chunks[index]
        correct = matches_expected(chunk, question)
        marker = "MATCH" if correct else "NO MATCH"

        print(f"\nRank: {rank} | Score: {score:.6f} | {marker}")
        print(f"Source: {chunk['source']}")
        print(f"Page: {chunk['page']}")
        print(f"Chunk ID: {chunk.get('chunk_id', 'N/A')}")
        print(f"Text:\n{chunk['text']}")


def main():
    for path in [
        EMBEDDINGS_FILE,
        CHUNKS_FILE,
        QUESTIONS_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    embeddings = np.load(EMBEDDINGS_FILE)
    chunks = load_jsonl(CHUNKS_FILE)
    questions = load_jsonl(QUESTIONS_FILE)

    if len(embeddings) != len(chunks):
        raise ValueError("Embeddings and chunks do not match.")

    tokenized_chunks = [
        tokenize(chunk["text"]) for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_chunks)
    model = SentenceTransformer(MODEL_NAME)

    failed_questions = []

    # Find questions where hybrid misses the correct page
    # within the first five results.
    for item in questions:
        dense_results = dense_ranking(
            item["question"], model, embeddings
        )
        bm25_results = bm25_ranking(
            item["question"], bm25, len(chunks)
        )
        hybrid_results = hybrid_ranking(
            dense_results, bm25_results, len(chunks)
        )

        hybrid_hit = any(
            matches_expected(chunks[index], item)
            for index, _ in hybrid_results[:TOP_K]
        )

        if not hybrid_hit:
            failed_questions.append(
                (item, dense_results, bm25_results, hybrid_results)
            )

    print(f"\nHybrid failures at Success@{TOP_K}: "
          f"{len(failed_questions)}")

    if not failed_questions:
        print("No failures found at this cutoff.")
        return

    for number, (
        item, dense_results, bm25_results, hybrid_results
    ) in enumerate(failed_questions, start=1):

        print("\n" + "=" * 75)
        print(f"FAILED QUESTION {number}")
        print(f"Question: {item['question']}")
        print(f"Expected source: {item['relevant_source']}")
        print(f"Expected pages: {item['relevant_pages']}")

        print_results(
            "DENSE", dense_results, chunks, item
        )
        print_results(
            "BM25", bm25_results, chunks, item
        )
        print_results(
            "HYBRID", hybrid_results, chunks, item
        )

        # Show where the expected page first appears in hybrid.
        expected_positions = [
            rank
            for rank, (index, _) in enumerate(
                hybrid_results, start=1
            )
            if matches_expected(chunks[index], item)
        ]

        if expected_positions:
            print(
                "\nFirst matching hybrid rank:",
                min(expected_positions)
            )
        else:
            print(
                "\nNo chunk matches the expected source/page "
                "anywhere in the current corpus."
            )


if __name__ == "__main__":
    main()
