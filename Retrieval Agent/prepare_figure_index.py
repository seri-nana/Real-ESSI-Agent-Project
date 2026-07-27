from __future__ import annotations

from pathlib import Path
import re

import fitz


PROJECT_FOLDER = Path(__file__).resolve().parent

PDF_PATH = (
    PROJECT_FOLDER
    / "documents"
    / "jeremic_lecture_notes_original.pdf"
)

OUTPUT_PATH = (
    PROJECT_FOLDER
    / "documents"
    / "jeremic_lecture_notes_figures.md"
)

REPORT_PATH = (
    PROJECT_FOLDER
    / "documents"
    / "figure_index_report.txt"
)


CAPTION_PATTERN = re.compile(
    r"""
    ^
    \s*
    (?P<type>Figure|Fig\.?|Table)
    \s*
    (?P<number>\d+(?:\.\d+)*)
    \s*
    (?:
        [:.\-]\s*
        |
        \s+
    )
    (?P<title>.*)
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def clean_line(line: str) -> str:
    line = line.replace("\x00", "")
    return " ".join(line.split())


def find_caption_entries(
    lines: list[str],
) -> list[tuple[int, str, str, str]]:
    """
    Returns:
        line index,
        content type,
        figure/table number,
        caption text
    """

    entries: list[tuple[int, str, str, str]] = []

    for index, line in enumerate(lines):
        cleaned = clean_line(line)

        match = CAPTION_PATTERN.match(cleaned)

        if not match:
            continue

        content_type = match.group("type")
        number = match.group("number")
        title = match.group("title").strip()

        # Sometimes the caption title wraps onto the next line.
        if not title and index + 1 < len(lines):
            next_line = clean_line(lines[index + 1])

            if next_line and not CAPTION_PATTERN.match(next_line):
                title = next_line

        entries.append(
            (
                index,
                content_type,
                number,
                title,
            )
        )

    return entries


def surrounding_text(
    lines: list[str],
    caption_index: int,
    before: int = 4,
    after: int = 6,
) -> str:
    """
    Preserve a small amount of verbatim source text surrounding
    each caption.
    """

    start = max(0, caption_index - before)
    end = min(len(lines), caption_index + after + 1)

    selected_lines = [
        line.rstrip()
        for line in lines[start:end]
        if line.strip()
    ]

    return "\n".join(selected_lines).strip()


def normalize_type(content_type: str) -> str:
    lowered = content_type.lower()

    if lowered.startswith("fig"):
        return "FIGURE"

    return "TABLE"


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found:\n{PDF_PATH}"
        )

    document = fitz.open(PDF_PATH)

    output_entries: list[str] = []
    pages_with_entries: set[int] = set()

    figure_count = 0
    table_count = 0

    print(f"Scanning {document.page_count} pages...")

    for page_index in range(document.page_count):
        page_number = page_index + 1
        page = document.load_page(page_index)

        raw_text = page.get_text(
            "text",
            sort=True,
        )

        lines = raw_text.splitlines()
        captions = find_caption_entries(lines)

        if not captions:
            continue

        pages_with_entries.add(page_number)

        for caption_index, raw_type, number, title in captions:
            content_type = normalize_type(raw_type)

            if content_type == "FIGURE":
                figure_count += 1
            else:
                table_count += 1

            caption_text = (
                f"{raw_type} {number}"
                + (f": {title}" if title else "")
            )

            context = surrounding_text(
                lines=lines,
                caption_index=caption_index,
            )

            entry = (
                "# JEREMIC LECTURE NOTES VISUAL INDEX\n\n"
                "CONTENT_TYPE: VISUAL_INDEX_ENTRY\n\n"
                f"VISUAL_TYPE: {content_type}\n\n"
                f"PDF_PAGE: {page_number}\n\n"
                f"ITEM_NUMBER: {number}\n\n"
                "DESCRIPTION_STATUS: NOT_VISUALLY_ANALYZED\n\n"
                "CAPTION_TEXT:\n"
                f"{caption_text}\n\n"
                "VERBATIM_SURROUNDING_TEXT:\n"
                "--- BEGIN SOURCE TEXT ---\n"
                f"{context}\n"
                "--- END SOURCE TEXT ---\n\n"
                "VISUAL_NOTE:\n"
                "The original PDF page contains a visual item associated "
                "with this caption. The diagram, graph, or table itself "
                "has not yet been interpreted."
            )

            output_entries.append(entry)

        print(
            f"Page {page_number}: "
            f"{len(captions)} caption(s)"
        )

    document.close()

    OUTPUT_PATH.write_text(
        "\n\n"
        + "\n\n========================================\n\n".join(
            output_entries
        )
        + "\n",
        encoding="utf-8",
    )

    report = [
        "FIGURE INDEX REPORT",
        "=" * 50,
        f"PDF pages scanned: 3012",
        f"Pages containing indexed captions: {len(pages_with_entries)}",
        f"Figure entries: {figure_count}",
        f"Table entries: {table_count}",
        f"Total index entries: {len(output_entries)}",
        "",
        f"Output file: {OUTPUT_PATH}",
    ]

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("\n".join(report))


if __name__ == "__main__":
    main()
