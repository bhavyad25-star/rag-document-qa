
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CHUNKS_FILE = DATA_DIR / "embedded_chunks.jsonl"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
QUESTIONS_FILE = DATA_DIR / "evaluation_questions.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60
CANDIDATE_COUNT = 20
K_VALUES = [3, 5, 10]


# --------------------------------------------------
# 2. LOAD JSONL FILES
# --------------------------------------------------

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


# --------------------------------------------------
# 3. TEXT TOKENIZATION FOR BM25
# --------------------------------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# --------------------------------------------------
# 4. DENSE RETRIEVAL
# --------------------------------------------------

def get_dense_order(question, model, embeddings):
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    scores = embeddings @ query_embedding
    return np.argsort(scores)[::-1].tolist()


# --------------------------------------------------
# 5. BM25 SPARSE RETRIEVAL
# --------------------------------------------------

def get_bm25_order(question, bm25):
    tokens = tokenize(question)
    scores = bm25.get_scores(tokens)
    return np.argsort(scores)[::-1].tolist()


# --------------------------------------------------
# 6. HYBRID RETRIEVAL USING RECIPROCAL RANK FUSION
# --------------------------------------------------

def fuse_rankings(dense_order, bm25_order, count):
    dense_ranks = np.empty(count, dtype=int)
    bm25_ranks = np.empty(count, dtype=int)

    for rank, index in enumerate(dense_order, start=1):
        dense_ranks[index] = rank

    for rank, index in enumerate(bm25_order, start=1):
        bm25_ranks[index] = rank

    # Combine dense and BM25 rankings.
    scores = (
        1.0 / (RRF_K + dense_ranks)
        + 1.0 / (RRF_K + bm25_ranks)
    )

    return np.argsort(scores)[::-1].tolist()


# --------------------------------------------------
# 7. RERANK THE TOP HYBRID CANDIDATES
# --------------------------------------------------

def rerank_candidates(question, hybrid_order, chunks, reranker):
    candidate_count = min(CANDIDATE_COUNT, len(hybrid_order))
    candidates = hybrid_order[:candidate_count]

    pairs = [
        (question, chunks[index]["text"])
        for index in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    candidate_scores = list(zip(candidates, scores))

    # Highest cross-encoder score first.
    candidate_scores.sort(
        key=lambda item: float(item[1]),
        reverse=True,
    )

    reranked_candidates = [
        index for index, score in candidate_scores
    ]

    # Keep all non-candidate results after the reranked candidates.
    candidate_set = set(candidates)

    remaining = [
        index
        for index in hybrid_order
        if index not in candidate_set
    ]

    return reranked_candidates + remaining


# --------------------------------------------------
# 8. EVALUATION: DOES TOP K CONTAIN A LABELLED CHUNK?
# --------------------------------------------------

def is_hit(ranking, question_item, chunks, k):
    expected_source = question_item["relevant_source"]
    expected_pages = question_item["relevant_pages"]

    for index in ranking[:k]:
        chunk = chunks[index]

        if (
            chunk.get("source") == expected_source
            and chunk.get("page") in expected_pages
        ):
            return True

    return False


# --------------------------------------------------
# 9. DEBUG EVALUATION LABELS
# --------------------------------------------------

def debug_evaluation_labels(questions):
    target_questions = {
        "What is the purpose of multi-head attention?",
        "How can retrieved passages help a language model answer knowledge-intensive questions?",
    }

    print("\nDEBUG: Evaluation labels loaded from disk")

    for item in questions:
        if item["question"] in target_questions:
            print("\nQuestion:", item["question"])
            print("Expected source:", repr(item["relevant_source"]))
            print("Expected pages:", repr(item["relevant_pages"]))


# --------------------------------------------------
# 10. MAIN EXPERIMENT
# --------------------------------------------------

def main():
    # Load data.
    chunks = read_jsonl(CHUNKS_FILE)
    questions = read_jsonl(QUESTIONS_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)

    print(f"Loaded {len(chunks)} chunks")
    print(f"Loaded {len(questions)} evaluation questions")

    # Check that the vectors and chunk metadata correspond.
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but "
            f"{len(embeddings)} embeddings. "
            "Run python src/embed_chunks.py again."
        )

    if not questions:
        raise ValueError("No evaluation questions were loaded.")

    # Display labels before evaluating retrieval.
    debug_evaluation_labels(questions)

    # Load the embedding model.
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Prepare BM25.
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_chunks)

    # Load the reranker.
    print(f"Loading reranker: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL)

    methods = {
        "Dense": [],
        "BM25": [],
        "Hybrid": [],
        "Hybrid+Rerank": [],
    }

    # Retrieve results for each question.
    for number, item in enumerate(questions, start=1):
        question = item["question"]

        dense_order = get_dense_order(
            question,
            embedding_model,
            embeddings,
        )

        bm25_order = get_bm25_order(
            question,
            bm25,
        )

        hybrid_order = fuse_rankings(
            dense_order,
            bm25_order,
            len(chunks),
        )

        reranked_order = rerank_candidates(
            question,
            hybrid_order,
            chunks,
            reranker,
        )

        methods["Dense"].append(dense_order)
        methods["BM25"].append(bm25_order)
        methods["Hybrid"].append(hybrid_order)
        methods["Hybrid+Rerank"].append(reranked_order)

        if number % 10 == 0 or number == len(questions):
            print(
                f"Processed {number}/{len(questions)} questions"
            )

    # --------------------------------------------------
    # 11. PRINT SUCCESS@K COMPARISON
    # --------------------------------------------------

    print("\nRetrieval comparison")
    print(
        f"{'Method':<18}"
        f"{'Success@3':>12}"
        f"{'Success@5':>12}"
        f"{'Success@10':>13}"
    )
    print("-" * 55)

    for method_name, rankings in methods.items():
        percentages = []

        for k in K_VALUES:
            hits = sum(
                is_hit(
                    rankings[i],
                    questions[i],
                    chunks,
                    k,
                )
                for i in range(len(questions))
            )

            percentage = hits / len(questions) * 100
            percentages.append(f"{percentage:.2f}%")

        print(
            f"{method_name:<18}"
            f"{percentages[0]:>12}"
            f"{percentages[1]:>12}"
            f"{percentages[2]:>13}"
        )

    # --------------------------------------------------
    # 12. INVESTIGATE DIFFERENCES AT TOP 3
    # --------------------------------------------------

    print("\nHybrid vs Hybrid+Rerank: questions with different @3 results")

    hybrid_rankings = methods["Hybrid"]
    reranked_rankings = methods["Hybrid+Rerank"]

    differences_found = False

    for i, item in enumerate(questions):
        hybrid_hit = is_hit(
            hybrid_rankings[i],
            item,
            chunks,
            3,
        )

        rerank_hit = is_hit(
            reranked_rankings[i],
            item,
            chunks,
            3,
        )

        if hybrid_hit != rerank_hit:
            differences_found = True

            print(f"\nQuestion: {item['question']}")
            print(f"Expected source: {item['relevant_source']}")
            print(f"Expected pages: {item['relevant_pages']}")
            print(f"Hybrid Success@3: {hybrid_hit}")
            print(f"Reranked Success@3: {rerank_hit}")

            print("\nReranked top 3:")

            for rank, index in enumerate(
                reranked_rankings[i][:3],
                start=1,
            ):
                chunk = chunks[index]

                print(
                    f"{rank}. {chunk.get('source')} "
                    f"page {chunk.get('page')} "
                    f"chunk={chunk.get('chunk_id')}"
                )

                print(
                    chunk["text"][:350].replace("\n", " ")
                )
                print()

    if not differences_found:
        print("No differences found at Success@3.")


if __name__ == "__main__":
    main()
