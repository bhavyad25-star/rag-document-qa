
import json
from pathlib import Path
import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "data" / "raw" / "papers"
EVAL_FILE = ROOT / "data" / "processed" / "evaluation_questions.jsonl"
OUTPUT_FILE = ROOT / "data" / "processed" / "evaluation_label_audit.txt"

with EVAL_FILE.open("r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

report = []

for i, item in enumerate(questions, start=1):
    source_txt = item["relevant_source"]
    pdf_name = source_txt.removesuffix(".txt") + ".pdf"
    pdf_path = PAPERS_DIR / pdf_name

    report.append("=" * 90)
    report.append(f"QUESTION {i}: {item['question']}")
    report.append(f"EXPECTED SOURCE: {source_txt}")
    report.append(f"EXPECTED PDF PAGES: {item['relevant_pages']}")

    if not pdf_path.exists():
        report.append(f"ERROR: PDF NOT FOUND: {pdf_path}")
        continue

    with fitz.open(pdf_path) as pdf:
        for page_num in item["relevant_pages"]:
            report.append("-" * 70)
            report.append(f"PDF PAGE {page_num}")

            if page_num < 1 or page_num > len(pdf):
                report.append(
                    f"ERROR: Page {page_num} is outside the PDF "
                    f"(it has {len(pdf)} pages)."
                )
                continue

            text = pdf[page_num - 1].get_text("text").strip()

            if not text:
                report.append(
                    "WARNING: No selectable text found on this page."
                )
            else:
                report.append(text[:2500])

OUTPUT_FILE.write_text("\n".join(report), encoding="utf-8")

print(f"Questions audited: {len(questions)}")
print(f"Audit report saved to: {OUTPUT_FILE}")
print("Open the report and check the question/page evidence.")
