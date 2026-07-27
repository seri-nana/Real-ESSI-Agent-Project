from __future__ import annotations

from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import re
import sys


# ============================================================
# 1. CONFIGURATION
# ============================================================

# The vector store containing the processed Real-ESSI sources.
VECTOR_STORE_ID = "vs_6a56bd35da488191a8efcdafc1e6272c"

# Ask the vector store for more results than we ultimately save.
# This gives the program room to remove duplicates while still
# preserving enough unique results.
SEARCH_RESULT_LIMIT = 20

# Maximum number of unique chunks saved for the next agent.
MAX_SAVED_CHUNKS = 10

# Main folder where all retrieval questions will be saved.
OUTPUT_ROOT = Path("questions")

# The two expected searchable source files.
VERBATIM_FILENAME = "jeremic_lecture_notes_verbatim.md"
FIGURES_FILENAME = "jeremic_lecture_notes_figures.md"

# Source ordering in the final output.
SOURCE_PRIORITY = {
    "VERBATIM_TEXT": 0,
    "VISUAL_INDEX": 1,
    "UNKNOWN": 99,
}


# ============================================================
# 2. CONNECT TO OPENAI
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print(
        "\nERROR: OPENAI_API_KEY was not found.\n"
        "Make sure your .env file contains:\n"
        "OPENAI_API_KEY=your-api-key"
    )
    sys.exit(1)

client = OpenAI(api_key=api_key)


# ============================================================
# 3. GET THE USER'S QUESTION
# ============================================================

def get_user_question() -> str:
    """
    Ask the user what they want to find in the Real-ESSI notes.

    This function only collects the question. It does not answer,
    summarize, or paraphrase the requested material.
    """

    print("\nREAL-ESSI NOTES RETRIEVAL AGENT")
    print("-" * 50)
    print(
        "Enter a question or describe the topic you want to find in the "
        "Real-ESSI lecture notes."
    )
    print(
        "Include important names, methods, equations, commands, figures, "
        "or concepts when possible."
    )
    print()
    print("Example:")
    print(
        "Find sections that explain the domain reduction method and how "
        "boundary conditions are applied."
    )
    print()

    while True:
        question = input("What would you like to find? ").strip()

        if not question:
            print("\nPlease enter a question or topic.\n")
            continue

        if len(question) < 5:
            print(
                "\nPlease provide a little more detail so the notes can be "
                "searched accurately.\n"
            )
            continue

        return question


# ============================================================
# 4. CREATE A NEW QUESTION FOLDER
# ============================================================

def get_next_question_number() -> int:
    """
    Find the next available numbered question folder.

    Example:
        question_001
        question_002
        question_003
    """

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    highest_number = 0

    for item in OUTPUT_ROOT.iterdir():
        if not item.is_dir():
            continue

        match = re.fullmatch(r"question_(\d+)", item.name)

        if match:
            folder_number = int(match.group(1))
            highest_number = max(highest_number, folder_number)

    return highest_number + 1


def create_output_folder() -> Path:
    """
    Create a new folder for the current retrieval.
    """

    question_number = get_next_question_number()

    output_folder = (
        OUTPUT_ROOT / f"question_{question_number:03d}"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=False,
    )

    return output_folder


# ============================================================
# 5. SEARCH THE VECTOR STORE
# ============================================================

def search_notes(question: str):
    """
    Search the Real-ESSI vector store.

    Query rewriting is enabled to improve retrieval wording while
    retaining the original user question in the saved output.
    """

    return client.vector_stores.search(
        vector_store_id=VECTOR_STORE_ID,
        query=question,
        rewrite_query=True,
        max_num_results=SEARCH_RESULT_LIMIT,
    )


# ============================================================
# 6. SOURCE CLASSIFICATION
# ============================================================

def classify_source(filename: str) -> str:
    """
    Classify a search result based on its filename.
    """

    normalized_filename = filename.strip().lower()

    if normalized_filename == VERBATIM_FILENAME.lower():
        return "VERBATIM_TEXT"

    if normalized_filename == FIGURES_FILENAME.lower():
        return "VISUAL_INDEX"

    return "UNKNOWN"


# ============================================================
# 7. METADATA EXTRACTION
# ============================================================

def extract_first_match(
    pattern: str,
    text: str,
) -> str | None:
    """
    Return the first captured regular-expression group.

    Returns None when the pattern is not present.
    """

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if not match:
        return None

    value = match.group(1).strip()

    return value or None


def extract_pdf_page(text: str) -> int | None:
    """
    Extract a physical PDF page number from a structured chunk.

    Expected marker:
        PDF_PAGE: 151
    """

    value = extract_first_match(
        r"^\s*PDF_PAGE\s*:\s*(\d+)\s*$",
        text,
    )

    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def extract_visual_type(text: str) -> str | None:
    """
    Extract the visual type from a visual-index entry.

    Expected examples:
        VISUAL_TYPE: FIGURE
        VISUAL_TYPE: TABLE
    """

    return extract_first_match(
        r"^\s*VISUAL_TYPE\s*:\s*([^\r\n]+)",
        text,
    )


def extract_item_number(text: str) -> str | None:
    """
    Extract a figure or table number.

    Expected example:
        ITEM_NUMBER: 102.20
    """

    return extract_first_match(
        r"^\s*ITEM_NUMBER\s*:\s*([^\r\n]+)",
        text,
    )


def extract_description_status(text: str) -> str | None:
    """
    Extract the visual-analysis status.

    Expected example:
        DESCRIPTION_STATUS: NOT_VISUALLY_ANALYZED
    """

    return extract_first_match(
        r"^\s*DESCRIPTION_STATUS\s*:\s*([^\r\n]+)",
        text,
    )


def extract_caption_text(text: str) -> str | None:
    """
    Extract the caption line from a visual-index entry.

    The generated figure-index format places the caption directly
    after the CAPTION_TEXT marker.
    """

    return extract_first_match(
        r"CAPTION_TEXT\s*:\s*\n\s*([^\r\n]+)",
        text,
    )


def build_chunk_metadata(
    filename: str,
    text: str,
) -> dict[str, Any]:
    """
    Build structured metadata without changing the source text.
    """

    source_type = classify_source(filename)

    metadata: dict[str, Any] = {
        "source_type": source_type,
        "pdf_page": extract_pdf_page(text),
        "visual_type": None,
        "item_number": None,
        "description_status": None,
        "caption_text": None,
        "requires_visual_analysis": False,
    }

    if source_type == "VISUAL_INDEX":
        description_status = extract_description_status(text)

        metadata.update(
            {
                "visual_type": extract_visual_type(text),
                "item_number": extract_item_number(text),
                "description_status": description_status,
                "caption_text": extract_caption_text(text),
                "requires_visual_analysis": (
                    description_status is not None
                    and description_status.upper()
                    == "NOT_VISUALLY_ANALYZED"
                ),
            }
        )

    return metadata


# ============================================================
# 8. DUPLICATE DETECTION
# ============================================================

def normalize_text_for_duplicate_check(text: str) -> str:
    """
    Normalize text only for duplicate comparison.

    The original text saved in the output is never changed.
    """

    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()

    # Remove repeated structural labels from the comparison key.
    normalized = re.sub(
        r"content_type\s*:\s*[a-z0-9_]+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )

    return normalized


def create_text_fingerprint(text: str) -> str:
    """
    Create a stable fingerprint for a retrieved chunk.
    """

    normalized = normalize_text_for_duplicate_check(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def create_partial_fingerprint(text: str) -> str:
    """
    Create a second fingerprint from the beginning of a chunk.

    This helps identify results that contain the same source passage
    with slightly different trailing content.
    """

    normalized = normalize_text_for_duplicate_check(text)

    comparison_text = normalized[:1000]

    return hashlib.sha256(
        comparison_text.encode("utf-8")
    ).hexdigest()


# ============================================================
# 9. EXTRACT AND PROCESS RETRIEVED CHUNKS
# ============================================================

def extract_chunks(search_results) -> list[dict[str, Any]]:
    """
    Copy text chunks from vector-store search results.

    This function:
        - preserves the exact retrieved text;
        - classifies the source;
        - extracts page and figure metadata;
        - removes duplicate chunks;
        - sorts verbatim text before visual-index entries;
        - limits the final saved output to MAX_SAVED_CHUNKS.
    """

    retrieved_chunks: list[dict[str, Any]] = []

    seen_full_fingerprints: set[str] = set()
    seen_partial_fingerprints: set[str] = set()

    for result_number, result in enumerate(
        search_results.data,
        start=1,
    ):
        filename = getattr(
            result,
            "filename",
            "Unknown file",
        )

        file_id = getattr(
            result,
            "file_id",
            None,
        )

        score = getattr(
            result,
            "score",
            None,
        )

        content_parts = getattr(
            result,
            "content",
            [],
        )

        for content_number, content_part in enumerate(
            content_parts,
            start=1,
        ):
            content_type = getattr(
                content_part,
                "type",
                None,
            )

            if content_type != "text":
                continue

            text = getattr(
                content_part,
                "text",
                "",
            )

            if not text or not text.strip():
                continue

            full_fingerprint = create_text_fingerprint(text)
            partial_fingerprint = create_partial_fingerprint(text)

            if full_fingerprint in seen_full_fingerprints:
                continue

            if partial_fingerprint in seen_partial_fingerprints:
                continue

            seen_full_fingerprints.add(full_fingerprint)
            seen_partial_fingerprints.add(partial_fingerprint)

            metadata = build_chunk_metadata(
                filename=filename,
                text=text,
            )

            retrieved_chunks.append(
                {
                    "original_result_number": result_number,
                    "content_number": content_number,
                    "file_id": file_id,
                    "filename": filename,
                    "source_type": metadata["source_type"],
                    "pdf_page": metadata["pdf_page"],
                    "visual_type": metadata["visual_type"],
                    "item_number": metadata["item_number"],
                    "description_status": (
                        metadata["description_status"]
                    ),
                    "caption_text": metadata["caption_text"],
                    "requires_visual_analysis": (
                        metadata["requires_visual_analysis"]
                    ),
                    "relevance_score": score,
                    "text": text,
                }
            )

    retrieved_chunks.sort(
        key=lambda chunk: (
            SOURCE_PRIORITY.get(
                chunk["source_type"],
                SOURCE_PRIORITY["UNKNOWN"],
            ),
            -(
                chunk["relevance_score"]
                if isinstance(
                    chunk["relevance_score"],
                    (int, float),
                )
                else 0.0
            ),
        )
    )

    return retrieved_chunks[:MAX_SAVED_CHUNKS]


# ============================================================
# 10. RETRIEVAL SUMMARY
# ============================================================

def build_retrieval_summary(
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize retrieval metadata without summarizing source content.
    """

    source_counts = {
        "VERBATIM_TEXT": 0,
        "VISUAL_INDEX": 0,
        "UNKNOWN": 0,
    }

    visual_pages: set[int] = set()
    visual_items: list[dict[str, Any]] = []

    for chunk in chunks:
        source_type = chunk["source_type"]

        source_counts[source_type] = (
            source_counts.get(source_type, 0) + 1
        )

        if source_type != "VISUAL_INDEX":
            continue

        pdf_page = chunk["pdf_page"]

        if isinstance(pdf_page, int):
            visual_pages.add(pdf_page)

        visual_items.append(
            {
                "pdf_page": pdf_page,
                "visual_type": chunk["visual_type"],
                "item_number": chunk["item_number"],
                "caption_text": chunk["caption_text"],
                "description_status": (
                    chunk["description_status"]
                ),
                "requires_visual_analysis": (
                    chunk["requires_visual_analysis"]
                ),
            }
        )

    return {
        "retrieved_chunk_count": len(chunks),
        "source_counts": source_counts,
        "visual_pages": sorted(visual_pages),
        "visual_items": visual_items,
        "visual_analysis_needed": any(
            item["requires_visual_analysis"]
            for item in visual_items
        ),
    }


# ============================================================
# 11. SAVE THE ORIGINAL QUESTION
# ============================================================

def save_question(
    question: str,
    output_folder: Path,
) -> Path:
    """
    Save the user's original question in its own text file.
    """

    question_file = output_folder / "question.txt"

    question_file.write_text(
        question,
        encoding="utf-8",
    )

    return question_file


# ============================================================
# 12. SAVE THE PLAIN-TEXT RETRIEVAL OUTPUT
# ============================================================

def save_text_output(
    question: str,
    chunks: list[dict[str, Any]],
    summary: dict[str, Any],
    output_folder: Path,
) -> Path:
    """
    Save the exact retrieved chunks in a readable text file.
    """

    text_file = output_folder / "retrieved_context.txt"

    source_counts = summary["source_counts"]
    visual_pages = summary["visual_pages"]

    visual_page_display = (
        ", ".join(str(page) for page in visual_pages)
        if visual_pages
        else "None"
    )

    output_sections = [
        "REAL-ESSI RETRIEVED CONTEXT",
        "=" * 70,
        f"Original user question: {question}",
        f"Number of retrieved chunks: {len(chunks)}",
        "",
        "RETRIEVAL SOURCE COUNTS",
        f"VERBATIM_TEXT: {source_counts.get('VERBATIM_TEXT', 0)}",
        f"VISUAL_INDEX: {source_counts.get('VISUAL_INDEX', 0)}",
        f"UNKNOWN: {source_counts.get('UNKNOWN', 0)}",
        "",
        f"Relevant visual PDF pages: {visual_page_display}",
        (
            "Visual analysis needed: "
            f"{summary['visual_analysis_needed']}"
        ),
        "",
        (
            "IMPORTANT: The source passages below were retrieved from the "
            "Real-ESSI notes. They have not been summarized, paraphrased, "
            "corrected, or rewritten."
        ),
        "=" * 70,
    ]

    for chunk_number, chunk in enumerate(
        chunks,
        start=1,
    ):
        score = chunk["relevance_score"]

        if isinstance(score, (int, float)):
            score_display = f"{score:.4f}"
        else:
            score_display = "Unavailable"

        pdf_page = (
            str(chunk["pdf_page"])
            if chunk["pdf_page"] is not None
            else "Unavailable"
        )

        output_sections.extend(
            [
                "",
                f"CHUNK {chunk_number}",
                "-" * 70,
                f"Source type: {chunk['source_type']}",
                f"Filename: {chunk['filename']}",
                f"File ID: {chunk['file_id']}",
                f"PDF page: {pdf_page}",
                f"Relevance score: {score_display}",
            ]
        )

        if chunk["source_type"] == "VISUAL_INDEX":
            output_sections.extend(
                [
                    (
                        "Visual type: "
                        f"{chunk['visual_type'] or 'Unavailable'}"
                    ),
                    (
                        "Item number: "
                        f"{chunk['item_number'] or 'Unavailable'}"
                    ),
                    (
                        "Description status: "
                        f"{chunk['description_status'] or 'Unavailable'}"
                    ),
                    (
                        "Requires visual analysis: "
                        f"{chunk['requires_visual_analysis']}"
                    ),
                ]
            )

            if chunk["caption_text"]:
                output_sections.append(
                    f"Caption: {chunk['caption_text']}"
                )

        output_sections.extend(
            [
                "--- BEGIN SOURCE TEXT ---",
                chunk["text"],
                "--- END SOURCE TEXT ---",
            ]
        )

    text_file.write_text(
        "\n".join(output_sections),
        encoding="utf-8",
    )

    return text_file


# ============================================================
# 13. SAVE THE STRUCTURED JSON OUTPUT
# ============================================================

def save_json_output(
    question: str,
    chunks: list[dict[str, Any]],
    summary: dict[str, Any],
    output_folder: Path,
) -> Path:
    """
    Save the question, metadata, and exact retrieved text as JSON.
    """

    json_file = output_folder / "retrieved_context.json"

    output_data = {
        "original_question": question,
        "vector_store_id": VECTOR_STORE_ID,
        "instructions_for_next_agent": [
            (
                "Use VERBATIM_TEXT chunks as exact source material from "
                "the lecture notes."
            ),
            (
                "Do not claim that VISUAL_INDEX chunks describe the visual "
                "content itself unless description_status indicates that "
                "visual analysis has been completed."
            ),
            (
                "When requires_visual_analysis is true, inspect the "
                "corresponding page in the original PDF before interpreting "
                "the diagram, figure, graph, or table."
            ),
            (
                "The retrieval agent has not summarized, paraphrased, "
                "corrected, or rewritten the retrieved source text."
            ),
        ],
        "retrieval_summary": summary,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    json_file.write_text(
        json.dumps(
            output_data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return json_file


# ============================================================
# 14. SAVE A VISUAL-HANDOFF FILE
# ============================================================

def save_visual_handoff(
    question: str,
    chunks: list[dict[str, Any]],
    output_folder: Path,
) -> Path | None:
    """
    Save a small JSON file containing only visual-index results.

    This is useful for a later vision agent. No visual interpretation
    is performed here.
    """

    visual_chunks = [
        chunk
        for chunk in chunks
        if chunk["source_type"] == "VISUAL_INDEX"
    ]

    if not visual_chunks:
        return None

    handoff_file = output_folder / "visual_handoff.json"

    handoff_data = {
        "original_question": question,
        "original_pdf_file_id": os.getenv(
            "REAL_ESSI_ORIGINAL_PDF_FILE_ID"
        ),
        "visual_result_count": len(visual_chunks),
        "instructions": (
            "Inspect only the listed PDF pages when visual interpretation "
            "is necessary. Do not replace or alter the verbatim lecture "
            "text retrieved by the retrieval agent."
        ),
        "visual_results": [
            {
                "pdf_page": chunk["pdf_page"],
                "visual_type": chunk["visual_type"],
                "item_number": chunk["item_number"],
                "caption_text": chunk["caption_text"],
                "description_status": (
                    chunk["description_status"]
                ),
                "requires_visual_analysis": (
                    chunk["requires_visual_analysis"]
                ),
                "relevance_score": (
                    chunk["relevance_score"]
                ),
                "source_text": chunk["text"],
            }
            for chunk in visual_chunks
        ],
    }

    handoff_file.write_text(
        json.dumps(
            handoff_data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return handoff_file


# ============================================================
# 15. REMOVE AN INCOMPLETE OUTPUT FOLDER
# ============================================================

def remove_output_folder(
    output_folder: Path,
) -> None:
    """
    Remove a newly created output folder after a save failure.
    """

    if not output_folder.exists():
        return

    for item in output_folder.iterdir():
        if item.is_file():
            item.unlink()

    output_folder.rmdir()


# ============================================================
# 16. PRINT THE TERMINAL SUMMARY
# ============================================================

def print_retrieval_summary(
    summary: dict[str, Any],
) -> None:
    """
    Print a metadata-only retrieval summary.
    """

    source_counts = summary["source_counts"]
    visual_pages = summary["visual_pages"]

    print("\nRETRIEVAL SUMMARY")
    print("-" * 50)
    print(
        f"Unique chunks saved: "
        f"{summary['retrieved_chunk_count']}"
    )
    print(
        "VERBATIM_TEXT chunks: "
        f"{source_counts.get('VERBATIM_TEXT', 0)}"
    )
    print(
        "VISUAL_INDEX chunks: "
        f"{source_counts.get('VISUAL_INDEX', 0)}"
    )
    print(
        "UNKNOWN chunks: "
        f"{source_counts.get('UNKNOWN', 0)}"
    )

    if visual_pages:
        print(
            "Relevant visual PDF pages: "
            + ", ".join(
                str(page)
                for page in visual_pages
            )
        )
    else:
        print("Relevant visual PDF pages: None")

    print(
        "Visual analysis needed: "
        f"{summary['visual_analysis_needed']}"
    )


# ============================================================
# 17. RUN THE RETRIEVAL WORKFLOW
# ============================================================

def main() -> None:
    """
    Run the complete retrieval-agent workflow.
    """

    question = get_user_question()

    print("\nSearching the Real-ESSI notes...")

    try:
        search_results = search_notes(question)

    except Exception as error:
        print("\nThe vector-store search failed.")
        print(f"Error details: {error}")
        sys.exit(1)

    chunks = extract_chunks(search_results)

    if not chunks:
        print(
            "\nNo relevant text chunks were found.\n"
            "Try asking the question with more specific terminology "
            "from the lecture notes."
        )
        sys.exit(0)

    summary = build_retrieval_summary(chunks)

    try:
        output_folder = create_output_folder()

        question_file = save_question(
            question=question,
            output_folder=output_folder,
        )

        text_file = save_text_output(
            question=question,
            chunks=chunks,
            summary=summary,
            output_folder=output_folder,
        )

        json_file = save_json_output(
            question=question,
            chunks=chunks,
            summary=summary,
            output_folder=output_folder,
        )

        visual_handoff_file = save_visual_handoff(
            question=question,
            chunks=chunks,
            output_folder=output_folder,
        )

    except Exception as error:
        if "output_folder" in locals():
            remove_output_folder(output_folder)

        print("\nThe retrieved files could not be saved.")
        print(f"Error details: {error}")
        sys.exit(1)

    print_retrieval_summary(summary)

    print("\nRetrieval complete.")
    print(f"Question folder: {output_folder.resolve()}")
    print(f"Question file: {question_file.resolve()}")
    print(f"Plain-text output: {text_file.resolve()}")
    print(f"Structured output: {json_file.resolve()}")

    if visual_handoff_file is not None:
        print(
            "Visual handoff: "
            f"{visual_handoff_file.resolve()}"
        )
    else:
        print("Visual handoff: Not created")

    print(
        "\nNo summary, generated answer, or visual interpretation "
        "was created. The retrieved source material is ready for "
        "the next agent."
    )


if __name__ == "__main__":
    main()
