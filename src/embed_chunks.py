
from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Find the project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# 2. Choose the embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # 3. Check that our input file exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find the chunk file: {INPUT_FILE}"
        )

    # 4. Read the chunks from the JSONL file
    chunks = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("The chunk file is empty.")

    # 5. Extract the text from every chunk
    texts = [chunk["text"] for chunk in chunks]

    print(f"Chunks loaded: {len(texts)}")

    # 6. Load the embedding model
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 7. Convert each text chunk into an embedding vector
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # 8. Save the vectors as a NumPy array
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vector_path = OUTPUT_DIR / "embeddings.npy"
    np.save(vector_path, embeddings)

    # 9. Save the corresponding text and metadata
    metadata_path = OUTPUT_DIR / "embedded_chunks.jsonl"

    with metadata_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    # 10. Check and report the results
    print("\nEmbedding generation complete!")
    print(f"Number of vectors: {embeddings.shape[0]}")
    print(f"Numbers per vector: {embeddings.shape[1]}")
    print(f"Vector file: {vector_path}")
    print(f"Metadata file: {metadata_path}")


if __name__ == "__main__":
    main()
