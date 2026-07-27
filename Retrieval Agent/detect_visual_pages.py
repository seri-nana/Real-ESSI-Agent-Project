from __future__ import annotations

from pathlib import Path
import csv
import re

import fitz  # PyMuPDF


PROJECT_FOLDER = Path(__file__).resolve().parent

PDF_PATH = (
    PROJECT_FOLDER
    / "documents"
    / "jeremic_lecture_notes_original.pdf"
)

REPORT_PATH = (
    PROJECT_FOLDER
    / "documents"
    / "visual_page_report.csv"
)

PAGE_LIST_PATH = (
    PROJECT_FOLDER
    / "documents"
    / "visual_pages.txt"
)


# Captions commonly used in the lecture notes.
FIGURE_PATTERN = re.compile(
    r"\b(?:figure|fig\.?)\s*\d+(?:\.\d+)*",
    re.IGNORECASE,
)

TABLE_PATTERN = re.compile(
    r"\btable\s*\d+(?:\.\d+)*",
    re.IGNORECASE,
)


def count_drawing_items(page: fitz.Page) -> tuple[int, int]:
    """
    Return:
        drawing_groups: Number of vector drawing objects.
        drawing_items: Number of individual paths, lines, curves, etc.
    """

    drawings = page.get_drawings()

    drawing_groups = len(drawings)
    drawing_items = sum(
        len(drawing.get("items", []))
        for drawing in drawings
    )

    return drawing_groups, drawing_items


def count_embedded_images(page: fitz.Page) -> int:
    """
    Count raster images referenced by the page.

    This may include logos and repeated background graphics.
    """

    return len(page.get_images(full=True))


def calculate_page_score(
    image_count: int,
    drawing_groups: int,
    drawing_items: int,
    figure_caption_count: int,
    table_caption_count: int,
) -> int:
    """
    Assign a rough visual-content score.

    The score is only used for finding candidate pages.
    It does not determine whether a figure is meaningful.
    """

    score = 0

    # Raster images are a strong signal.
    if image_count > 0:
        score += 3

    # Figure captions are a very strong signal.
    score += figure_caption_count * 5

    # Table captions are also useful.
    score += table_caption_count * 4

    # Vector drawings can represent graphs, diagrams, borders,
    # equation symbols, page decorations, or underlines.
    if drawing_groups >= 1:
        score += 1

    if drawing_groups >= 5:
        score += 2

    if drawing_items >= 20:
        score += 2

    if drawing_items >= 50:
        score += 2

    return score


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            "Could not find the PDF:\n"
            f"{PDF_PATH}"
        )

    document = fitz.open(PDF_PATH)

    rows: list[dict[str, object]] = []
    candidate_pages: list[int] = []

    print(f"PDF: {PDF_PATH.name}")
    print(f"Pages: {document.page_count}")
    print("-" * 78)

    for page_index in range(document.page_count):
        page_number = page_index + 1
        page = document.load_page(page_index)

        text = page.get_text("text", sort=True)

        figure_matches = FIGURE_PATTERN.findall(text)
        table_matches = TABLE_PATTERN.findall(text)

        image_count = count_embedded_images(page)

        drawing_groups, drawing_items = count_drawing_items(page)

        score = calculate_page_score(
            image_count=image_count,
            drawing_groups=drawing_groups,
            drawing_items=drawing_items,
            figure_caption_count=len(figure_matches),
            table_caption_count=len(table_matches),
        )

        # A score of 5 usually means there is at least a figure caption
        # or a significant amount of visual material.
        is_candidate = score >= 5

        if is_candidate:
            candidate_pages.append(page_number)

        rows.append(
            {
                "pdf_page": page_number,
                "visual_score": score,
                "candidate": is_candidate,
                "embedded_images": image_count,
                "drawing_groups": drawing_groups,
                "drawing_items": drawing_items,
                "figure_captions": len(figure_matches),
                "table_captions": len(table_matches),
                "matched_figure_text": " | ".join(figure_matches),
                "matched_table_text": " | ".join(table_matches),
                "extracted_characters": len(text.strip()),
            }
        )

        if is_candidate:
            print(
                f"Page {page_number:4} | "
                f"score={score:2} | "
                f"images={image_count:2} | "
                f"drawing groups={drawing_groups:3} | "
                f"drawing items={drawing_items:4} | "
                f"figures={len(figure_matches):2} | "
                f"tables={len(table_matches):2}"
            )

    document.close()

    with REPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    PAGE_LIST_PATH.write_text(
        "\n".join(str(page) for page in candidate_pages),
        encoding="utf-8",
    )

    print("-" * 78)
    print(f"Candidate visual pages: {len(candidate_pages)}")
    print(f"CSV report:\n{REPORT_PATH}")
    print(f"Page-number list:\n{PAGE_LIST_PATH}")


if __name__ == "__main__":
    main()
