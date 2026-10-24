
from pathlib import Path
import json
import fitz  # PyMuPDF

# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define where the original PDFs are stored.
PAPERS_DIR = PROJECT_ROOT / "data" / "raw" / "papers"

# Define where extracted text and reports will be saved.
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Create the output folder if it does not exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_pdf(pdf_path):
    """Extract text page by page and inspect PDF quality."""

    document = fitz.open(pdf_path)

    page_records = []
    total_characters = 0
    low_text_pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        character_count = len(text)
        total_characters += character_count

        if character_count < 80:
            low_text_pages.append(page_number)

        page_records.append({
            "page": page_number,
            "text": text,
            "characters": character_count
        })

    result = {
        "filename": pdf_path.name,
        "total_pages": len(document),
        "total_characters": total_characters,
        "average_characters_per_page": round(
            total_characters / len(document), 2
        ) if len(document) else 0,
        "low_text_pages": low_text_pages,
        "pages": page_records
    }

    document.close()
    return result


def main():
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in: {PAPERS_DIR}")
        print("Place your research papers in that folder.")
        return

    all_results = []

    for pdf_path in pdf_files:
        try:
            result = inspect_pdf(pdf_path)
            all_results.append(result)

            print(f"\nPaper: {result['filename']}")
            print(f"Pages: {result['total_pages']}")
            print(f"Extracted characters: {result['total_characters']}")
            print(
                "Average characters per page: "
                f"{result['average_characters_per_page']}"
            )

            if result["low_text_pages"]:
                print(
                    "Pages needing inspection: "
                    f"{result['low_text_pages']}"
                )
            else:
                print("No obviously low-text pages detected.")

            text_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"

            with text_path.open("w", encoding="utf-8") as file:
                for page in result["pages"]:
                    file.write(
                        f"\n\n--- PAGE {page['page']} ---\n\n"
                    )
                    file.write(page["text"])

            print(f"Extracted text saved to: {text_path}")

        except Exception as error:
            print(f"\nCould not process {pdf_path.name}: {error}")

    report = [
        {
            "filename": result["filename"],
            "total_pages": result["total_pages"],
            "total_characters": result["total_characters"],
            "average_characters_per_page":
                result["average_characters_per_page"],
            "low_text_pages": result["low_text_pages"]
        }
        for result in all_results
    ]

    report_path = OUTPUT_DIR / "pdf_inspection_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(f"\nInspection report saved to: {report_path}")


if __name__ == "__main__":
    main()
