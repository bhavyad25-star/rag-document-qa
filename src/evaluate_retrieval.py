
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Find the project folder and data files.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "embedded_chunks.jsonl"
QUESTIONS_FILE = PROJECT_ROOT / "data" / "processed" / "evaluation_questions.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


# 2. Load JSONL data: one JSON object per line.
def load_jsonl(file_path):
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


# 3. Retrieve the top-k most similar chunks for a question.
def retrieve(question, model, embeddings, chunks):
    question_vector = model.encode(
        question, normalize_embeddings=True
    )

    scores = embeddings @ question_vector
    best_indices = np.argsort(scores)[::-1][:TOP_K]

    return [
        {
            "source": chunks[int(i)]["source"],
            "page": chunks[int(i)]["page"],
            "score": float(scores[i]),
        }
        for i in best_indices
    ]


# 4. Check whether retrieval found a labeled relevant page.
def is_hit(results, expected):
    relevant_source = expected["relevant_source"]
    relevant_pages = expected["relevant_pages"]

    for result in results:
        if (
            result["source"] == relevant_source
            and result["page"] in relevant_pages
        ):
            return True

    return False


# 5. Run the evaluation.
def main():
    for path in [EMBEDDINGS_FILE, CHUNKS_FILE, QUESTIONS_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    embeddings = np.load(EMBEDDINGS_FILE)
    chunks = load_jsonl(CHUNKS_FILE)
    questions = load_jsonl(QUESTIONS_FILE)

    if len(embeddings) != len(chunks):
        raise ValueError("Embeddings and chunks do not match.")

    if not questions:
        raise ValueError("The evaluation question file is empty.")

    model = SentenceTransformer(MODEL_NAME)

    hits = 0

    for number, item in enumerate(questions, start=1):
        results = retrieve(
            item["question"], model, embeddings, chunks
        )

        hit = is_hit(results, item)
        hits += int(hit)

        print(f"\nQuestion {number}: {item['question']}")
        print(f"Relevant page found in top {TOP_K}: {hit}")

        for rank, result in enumerate(results, start=1):
            print(
                f"  {rank}. {result['source']}, "
                f"page {result['page']}, "
                f"score={result['score']:.4f}"
            )

    success_rate = hits / len(questions) * 100

    print("\n--- EVALUATION SUMMARY ---")
    print(f"Questions tested: {len(questions)}")
    print(f"Successful retrievals: {hits}")
    print(f"Success@{TOP_K}: {success_rate:.2f}%")


if __name__ == "__main__":
    main()
