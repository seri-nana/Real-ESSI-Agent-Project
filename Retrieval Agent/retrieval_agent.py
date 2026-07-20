from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import json
import os
import re
import sys


# ============================================================
# 1. CONFIGURATION
# ============================================================

# The vector store containing the Real-ESSI lecture notes.
VECTOR_STORE_ID = "vs_6a56bd35da488191a8efcdafc1e6272c"

# Maximum number of relevant chunks returned by the search.
MAX_RESULTS = 10

# Main folder where all retrieval questions will be saved.
OUTPUT_ROOT = Path("questions")


# ============================================================
# 2. CONNECT TO OPENAI
# ============================================================

# Load variables from the .env file.
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

    The function only collects the question.
    It does not answer or summarize it.
    """

    print("\nREAL-ESSI NOTES RETRIEVAL AGENT")
    print("-" * 40)
    print(
        "Enter a question or describe the topic you want to find in the "
        "Real-ESSI lecture notes."
    )
    print(
        "Include important names, methods, equations, commands, or concepts "
        "when possible."
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
    Find the next available question number.

    For example, if question_001 and question_002 already exist,
    this function returns 3.
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
    Create a new numbered folder for the current retrieval.

    Example:
        questions/question_001
        questions/question_002
    """

    question_number = get_next_question_number()

    output_folder = (
        OUTPUT_ROOT / f"question_{question_number:03d}"
    )

    output_folder.mkdir(parents=True, exist_ok=False)

    return output_folder


# ============================================================
# 5. SEARCH THE VECTOR STORE
# ============================================================

def search_notes(question: str):
    """
    Search the Real-ESSI vector store for relevant text chunks.

    rewrite_query=True allows the search system to improve the search
    wording while keeping the original user question unchanged.

    This function does not call a language model to write an answer.
    """

    return client.vector_stores.search(
        vector_store_id=VECTOR_STORE_ID,
        query=question,
        rewrite_query=True,
        max_num_results=MAX_RESULTS,
    )


# ============================================================
# 6. EXTRACT THE RETRIEVED CHUNKS
# ============================================================

def extract_chunks(search_results) -> list[dict]:
    """
    Copy text chunks from the vector-store search response.

    The retrieved text is not summarized, paraphrased, corrected,
    shortened, or otherwise rewritten.
    """

    retrieved_chunks = []

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

            retrieved_chunks.append(
                {
                    "result_number": result_number,
                    "content_number": content_number,
                    "file_id": file_id,
                    "filename": filename,
                    "relevance_score": score,
                    "text": text,
                }
            )

    return retrieved_chunks


# ============================================================
# 7. SAVE THE ORIGINAL QUESTION
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
# 8. SAVE THE PLAIN-TEXT RETRIEVAL OUTPUT
# ============================================================

def save_text_output(
    question: str,
    chunks: list[dict],
    output_folder: Path,
) -> Path:
    """
    Save all retrieved chunks in a readable plain-text file.

    Text between BEGIN SOURCE TEXT and END SOURCE TEXT comes from
    the vector-store search results and is not summarized.
    """

    text_file = output_folder / "retrieved_context.txt"

    output_sections = [
        "REAL-ESSI RETRIEVED CONTEXT",
        "=" * 70,
        f"Original user question: {question}",
        f"Number of retrieved chunks: {len(chunks)}",
        "",
        (
            "IMPORTANT: The source passages below were retrieved from the "
            "Real-ESSI notes. They have not been summarized or paraphrased."
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

        output_sections.extend(
            [
                "",
                f"CHUNK {chunk_number}",
                f"Filename: {chunk['filename']}",
                f"File ID: {chunk['file_id']}",
                f"Relevance score: {score_display}",
                "--- BEGIN SOURCE TEXT ---",
                chunk["text"],
                "--- END SOURCE TEXT ---",
            ]
        )

    final_output = "\n".join(output_sections)

    text_file.write_text(
        final_output,
        encoding="utf-8",
    )

    return text_file


# ============================================================
# 9. SAVE THE STRUCTURED JSON OUTPUT
# ============================================================

def save_json_output(
    question: str,
    chunks: list[dict],
    output_folder: Path,
) -> Path:
    """
    Save the question, retrieved text, and metadata as JSON.

    This format is useful for the next Python agent because each
    chunk remains separate and easy to access.
    """

    json_file = output_folder / "retrieved_context.json"

    output_data = {
        "original_question": question,
        "instructions_for_next_agent": (
            "Use the retrieved source text as context. "
            "The retrieval agent has not summarized or paraphrased it."
        ),
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
# 10. REMOVE AN EMPTY OUTPUT FOLDER
# ============================================================

def remove_empty_output_folder(
    output_folder: Path,
) -> None:
    """
    Delete the newly created folder when retrieval fails or returns
    no usable chunks.
    """

    if not output_folder.exists():
        return

    for item in output_folder.iterdir():
        if item.is_file():
            item.unlink()

    output_folder.rmdir()


# ============================================================
# 11. RUN THE RETRIEVAL WORKFLOW
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
            "Try asking the question with more specific terms from the notes."
        )
        sys.exit(0)

    try:
        output_folder = create_output_folder()

        question_file = save_question(
            question,
            output_folder,
        )

        text_file = save_text_output(
            question,
            chunks,
            output_folder,
        )

        json_file = save_json_output(
            question,
            chunks,
            output_folder,
        )

    except Exception as error:
        if "output_folder" in locals():
            remove_empty_output_folder(output_folder)

        print("\nThe retrieved files could not be saved.")
        print(f"Error details: {error}")
        sys.exit(1)

    print("\nRetrieval complete.")
    print(f"Retrieved chunks: {len(chunks)}")
    print(f"Question folder: {output_folder.resolve()}")
    print(f"Question file: {question_file.resolve()}")
    print(f"Plain-text output: {text_file.resolve()}")
    print(f"Structured output: {json_file.resolve()}")
    print(
        "\nNo summary or generated answer was created. "
        "The retrieved text is ready for the next agent."
    )


if __name__ == "__main__":
    main()
