import json
from pathlib import Path

# Locate the evaluation file from the project root.
file_path = Path("data/processed/evaluation_questions.jsonl")

# Store all valid questions here.
questions = []

# Read the file one line at a time.
with file_path.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):

        # Ignore empty lines.
        if not line.strip():
            continue

        try:
            # Convert this line from JSON text into a Python dictionary.
            question = json.loads(line)

        except json.JSONDecodeError as error:
            print(f"Invalid JSON on line {line_number}: {error}")
            continue

        # Check that the required fields exist.
        required_fields = {
            "question",
            "relevant_source",
            "relevant_pages",
        }

        if not required_fields.issubset(question):
            print(f"Missing fields on line {line_number}")
            continue

        # Check the expected data types.
        if (
            not isinstance(question["question"], str)
            or not isinstance(question["relevant_source"], str)
            or not isinstance(question["relevant_pages"], list)
        ):
            print(f"Incorrect field types on line {line_number}")
            continue

        questions.append(question)

# Print a summary.
print("\nValidation complete.")
print("Valid questions:", len(questions))
print("Unique source files:", {
    question["relevant_source"] for question in questions
})