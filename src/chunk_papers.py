
import json
import re
from pathlib import Path

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "papers"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

# Improved first candidate for retrieval experiments
TARGET_CHARS = 1000
OVERLAP_CHARS = 150


def clean_text(text):
    """Clean PDF extraction artifacts while preserving paragraph boundaries."""
    text = text.replace("\u00ad", "")
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long_sentence(sentence, max_chars):
    """Split unusually long sentences at word boundaries."""
    words = sentence.split()
    pieces = []
    current = ""

    for word in words:
        # Handle a single exceptionally long token as a last resort.
        if len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""

            pieces.extend(
                word[i:i + max_chars]
                for i in range(0, len(word), max_chars)
            )
            continue

        candidate = f"{current} {word}".strip()

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = word

    if current:
        pieces.append(current)

    return pieces


def sentences_from_page(text, max_chars):
    """Extract sentence-sized units while preserving paragraph boundaries."""
    text = clean_text(text)
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    sentences = []

    for paragraph in paragraphs:
        # Split at likely sentence endings, including common PDF spacing.
        parts = re.split(r"(?<=[.!?])\s+", paragraph)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part) <= max_chars:
                sentences.append(part)
            else:
                sentences.extend(
                    split_long_sentence(part, max_chars)
                )

    return sentences


def make_page_chunks(sentences, max_chars, overlap_chars):
    """Pack sentences into chunks and overlap whole sentences where possible."""
    chunks = []
    current = []

    def current_text():
        return " ".join(current).strip()

    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()

        if len(candidate) <= max_chars:
            current.append(sentence)
            continue

        if current:
            chunks.append(current_text())

        # Keep complete trailing sentences as overlap.
        overlap = []
        overlap_length = 0

        for previous_sentence in reversed(current):
            extra_length = len(previous_sentence) + (
                1 if overlap else 0
            )

            if overlap_length + extra_length > overlap_chars:
                break

            overlap.insert(0, previous_sentence)
            overlap_length += extra_length

        # Avoid overlap leaving no room for the next sentence.
        while overlap and len(
            " ".join(overlap + [sentence])
        ) > max_chars:
            overlap.pop(0)

        current = overlap

        # Long sentences have already been split, but retain a
        # defensive fallback to prevent oversized chunks.
        if len(sentence) > max_chars:
            sentence = sentence[:max_chars]

        current.append(sentence)

    if current:
        final_chunk = current_text()
        if final_chunk:
            chunks.append(final_chunk)

    return chunks


def main():
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_DIR}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        with fitz.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                page_text = page.get_text("text")

                sentences = sentences_from_page(
                    page_text,
                    TARGET_CHARS,
                )

                page_chunks = make_page_chunks(
                    sentences,
                    TARGET_CHARS,
                    OVERLAP_CHARS,
                )

                for chunk_index, text in enumerate(
                    page_chunks, start=1
                ):
                    if not text.strip():
                        continue

                    all_chunks.append({
                        "chunk_id": (
                            f"{pdf_path.stem}_p{page_number:03d}"
                            f"_c{chunk_index:03d}"
                        ),
                        "source": f"{pdf_path.stem}.txt",
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "text": text,
                    })

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    lengths = [len(chunk["text"]) for chunk in all_chunks]

    print("\nChunking complete!")
    print(f"PDFs processed: {len(pdf_files)}")
    print(f"Total chunks: {len(all_chunks)}")

    if lengths:
        print(f"Average chunk length: {sum(lengths) / len(lengths):.1f}")
        print(f"Shortest chunk: {min(lengths)} characters")
        print(f"Longest chunk: {max(lengths)} characters")

    print(f"Target chunk size: {TARGET_CHARS}")
    print(f"Sentence-boundary overlap target: {OVERLAP_CHARS}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

