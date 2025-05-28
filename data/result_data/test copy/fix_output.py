#!/usr/bin/env python3
import json
import os


def remove_rerank_fields(file_path):
    """
    Removes 'rerank_score' and 'rerank_position' fields from all snippets in the JSON file.

    Args:
        file_path: Path to the JSON file to modify
    """
    print(f"Processing file: {file_path}")

    # Read JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Track number of fields removed
    removed_fields = 0

    # Iterate through all questions
    for question in data.get("questions", []):
        # Iterate through snippets in each question
        for snippet in question.get("snippets", []):
            # Remove the fields if they exist
            if "rerank_score" in snippet:
                del snippet["rerank_score"]
                removed_fields += 1
            if "rerank_position" in snippet:
                del snippet["rerank_position"]
                removed_fields += 1

    # Write the modified data back to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Successfully removed {removed_fields} fields from the file")
    print(f"Modified file saved to: {file_path}")


if __name__ == "__main__":
    file_path = (
        "data/result_data/test/TEST_results_PubMedBERT_2025-05-08_02-34-30.json"
    )
    if os.path.exists(file_path):
        remove_rerank_fields(file_path)
    else:
        print(f"Error: File not found at {file_path}")
