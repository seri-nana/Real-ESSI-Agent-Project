from __future__ import annotations

from pathlib import Path
import re

import fitz  # PyMuPDF


PROJECT_FOLDER = Path(__file__).resolve().parent

PDF_PATH = (
    PROJECT_FOLDER
    / "documents"
    / "jeremic_lecture_notes_original.pdf"
)

OUTPUT_PATH = (
    PROJECT_FOLDER
    / "documents"
    / "jeremic_lecture_notes_verbatim.md"
)

REPORT_PATH = (
    PROJECT_FOLDER
    / "documents"
    / "verbatim_extraction_report.txt"
)


def clean_extracted_text(text: str) -> str:
    """
    Clean characters that interfere with storage while preserving the
    wording extracted from the source PDF.

    This function does not summarize or rewrite the lecture notes.
    """

    # Remove null bytes and characters that commonly break text files.
    text = text.replace("\x00", "")

    # Normalize Windows and old Mac line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing whitespace without changing the words.
    lines = [line.rstrip() for line in text.splitlines()]

    # Reduce runs of more than three blank lines.
    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"\n{4,}", "\n\n\n", cleaned_text)

    return cleaned_text.strip()


def extract_verbatim_notes() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            "The original PDF could not be found.\n"
            f"Expected location:\n{PDF_PATH}"
        )

    document = fitz.open(PDF_PATH)

    output_sections: list[str] = []
    report_lines: list[str] = []

    total_characters = 0
    low_text_pages: list[int] = []
    empty_pages: list[int] = []

    print(f"PDF: {PDF_PATH.name}")
    print(f"Total pages: {document.page_count}")
    print("-" * 60)

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        page_number = page_index + 1

        # "sort=True" attempts to place text in normal reading order.
        raw_text = page.get_text("text", sort=True)
        cleaned_text = clean_extracted_text(raw_text)

        character_count = len(cleaned_text)
        total_characters += character_count

        if character_count == 0:
            status = "EMPTY"
            empty_pages.append(page_number)
        elif character_count < 200:
            status = "LOW TEXT"
            low_text_pages.append(page_number)
        else:
            status = "OK"

        print(
            f"Page {page_number:4}: "
            f"{character_count:6} characters | {status}"
        )

        report_lines.append(
            f"Page {page_number}: "
            f"{character_count} characters | {status}"
        )

        if not cleaned_text:
            cleaned_text = (
                "[No embedded text was extracted from this page.]"
            )

        section = (
            f"# SOURCE: JEREMIC LECTURE NOTES\n\n"
            f"CONTENT_TYPE: VERBATIM_PDF_TEXT\n\n"
            f"PDF_PAGE: {page_number}\n\n"
            f"--- BEGIN EXTRACTED PAGE TEXT ---\n\n"
            f"{cleaned_text}\n\n"
            f"--- END EXTRACTED PAGE TEXT ---"
        )

        output_sections.append(section)

    document.close()

    OUTPUT_PATH.write_text(
        "\n\n"
        + "\n\n========================================\n\n".join(
            output_sections
        )
        + "\n",
        encoding="utf-8",
    )

    report_summary = [
        "JEREMIC LECTURE NOTES EXTRACTION REPORT",
        "=" * 50,
        f"Original PDF: {PDF_PATH}",
        f"Output file: {OUTPUT_PATH}",
        f"Total pages: {len(output_sections)}",
        f"Total extracted characters: {total_characters}",
        "",
        "Pages with fewer than 200 extracted characters:",
        str(low_text_pages) if low_text_pages else "None",
        "",
        "Pages with no extracted text:",
        str(empty_pages) if empty_pages else "None",
        "",
        "PER-PAGE RESULTS",
        "-" * 50,
        *report_lines,
    ]

    REPORT_PATH.write_text(
        "\n".join(report_summary),
        encoding="utf-8",
    )

    print("-" * 60)
    print("Extraction complete.")
    print(f"Verbatim output:\n{OUTPUT_PATH}")
    print(f"Extraction report:\n{REPORT_PATH}")
    print()
    print(f"Low-text pages: {len(low_text_pages)}")
    print(f"Empty pages: {len(empty_pages)}")


if __name__ == "__main__":
    extract_verbatim_notes()
