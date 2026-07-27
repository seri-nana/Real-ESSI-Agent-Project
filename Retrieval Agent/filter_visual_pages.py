from __future__ import annotations

from pathlib import Path
import csv


PROJECT_FOLDER = Path(__file__).resolve().parent

INPUT_REPORT = (
    PROJECT_FOLDER
    / "documents"
    / "visual_page_report.csv"
)

HIGH_CONFIDENCE_OUTPUT = (
    PROJECT_FOLDER
    / "documents"
    / "high_confidence_visual_pages.txt"
)

REVIEW_OUTPUT = (
    PROJECT_FOLDER
    / "documents"
    / "possible_uncaptioned_visual_pages.txt"
)

SUMMARY_OUTPUT = (
    PROJECT_FOLDER
    / "documents"
    / "visual_filter_summary.txt"
)


def to_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def main() -> None:
    if not INPUT_REPORT.exists():
        raise FileNotFoundError(
            "Could not find the visual-page report:\n"
            f"{INPUT_REPORT}"
        )

    high_confidence_pages: list[int] = []
    possible_uncaptioned_pages: list[int] = []

    figure_pages = 0
    table_pages = 0
    raster_image_pages = 0

    with INPUT_REPORT.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            page_number = to_int(row.get("pdf_page"))

            figure_captions = to_int(
                row.get("figure_captions")
            )

            table_captions = to_int(
                row.get("table_captions")
            )

            embedded_images = to_int(
                row.get("embedded_images")
            )

            drawing_groups = to_int(
                row.get("drawing_groups")
            )

            drawing_items = to_int(
                row.get("drawing_items")
            )

            if figure_captions > 0:
                figure_pages += 1

            if table_captions > 0:
                table_pages += 1

            if embedded_images > 0:
                raster_image_pages += 1

            # Strongest signals:
            # the page explicitly contains a Figure or Table caption.
            if figure_captions > 0 or table_captions > 0:
                high_confidence_pages.append(page_number)
                continue

            # Keep unusual, heavily drawn pages in a separate review list.
            # Do not automatically send these to the figure pipeline.
            if (
                drawing_groups >= 15
                and drawing_items >= 100
            ):
                possible_uncaptioned_pages.append(page_number)

    HIGH_CONFIDENCE_OUTPUT.write_text(
        "\n".join(
            str(page)
            for page in high_confidence_pages
        ),
        encoding="utf-8",
    )

    REVIEW_OUTPUT.write_text(
        "\n".join(
            str(page)
            for page in possible_uncaptioned_pages
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "VISUAL-PAGE FILTER SUMMARY",
        "=" * 50,
        f"Pages containing figure captions: {figure_pages}",
        f"Pages containing table captions: {table_pages}",
        f"Pages containing embedded raster images: {raster_image_pages}",
        "",
        (
            "High-confidence visual pages: "
            f"{len(high_confidence_pages)}"
        ),
        (
            "Possible uncaptioned visual pages: "
            f"{len(possible_uncaptioned_pages)}"
        ),
        "",
        f"High-confidence list: {HIGH_CONFIDENCE_OUTPUT}",
        f"Review list: {REVIEW_OUTPUT}",
    ]

    SUMMARY_OUTPUT.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
