from __future__ import annotations

from pathlib import Path
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_FOLDER = Path(__file__).resolve().parent
DOCUMENTS_FOLDER = PROJECT_FOLDER / "documents"

SOURCE_FILES = [
    DOCUMENTS_FOLDER / "jeremic_lecture_notes_original.pdf",
    DOCUMENTS_FOLDER / "jeremic_lecture_notes_verbatim.md",
    DOCUMENTS_FOLDER / "jeremic_lecture_notes_figures.md",
]

# Replace this only if your current vector-store ID is different.
VECTOR_STORE_ID = "vs_6a56bd35da488191a8efcdafc1e6272c"

POLL_INTERVAL_SECONDS = 5
MAX_WAIT_SECONDS = 60 * 60


def validate_source_files() -> None:
    missing_files = [
        path
        for path in SOURCE_FILES
        if not path.exists()
    ]

    if missing_files:
        formatted = "\n".join(
            f"- {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "The following source files could not be found:\n"
            f"{formatted}"
        )

    print("Source files found:")

    for path in SOURCE_FILES:
        size_mb = path.stat().st_size / (1024 * 1024)

        print(
            f"  {path.name}: "
            f"{size_mb:.2f} MB"
        )


def wait_for_vector_store_file(
    client: OpenAI,
    vector_store_id: str,
    file_id: str,
    filename: str,
) -> str:
    """
    Wait until one uploaded file has either completed or failed.
    """

    start_time = time.monotonic()

    while True:
        vector_store_file = client.vector_stores.files.retrieve(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )

        status = vector_store_file.status

        elapsed = int(time.monotonic() - start_time)

        print(
            f"  {filename}: "
            f"{status} "
            f"({elapsed} seconds)"
        )

        if status == "completed":
            return status

        if status in {"failed", "cancelled"}:
            print(
                f"  Processing error for {filename}: "
                f"{vector_store_file.last_error}"
            )
            return status

        if elapsed >= MAX_WAIT_SECONDS:
            print(
                f"  Timed out waiting for {filename}."
            )
            return "timed_out"

        time.sleep(POLL_INTERVAL_SECONDS)


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found in the .env file."
        )

    validate_source_files()

    client = OpenAI(api_key=api_key)

    # Confirm that the vector store exists before uploading anything.
    vector_store = client.vector_stores.retrieve(
        vector_store_id=VECTOR_STORE_ID
    )

    print()
    print("VECTOR STORE")
    print("-" * 60)
    print(f"Name: {vector_store.name}")
    print(f"ID:   {vector_store.id}")

    uploaded_files: list[dict[str, str]] = []

    print()
    print("UPLOADING SOURCE FILES")
    print("-" * 60)

    for source_path in SOURCE_FILES:
        print(f"\nUploading: {source_path.name}")

        with source_path.open("rb") as file_stream:
            uploaded_file = client.files.create(
                file=file_stream,
                purpose="assistants",
            )

        print(f"OpenAI file ID: {uploaded_file.id}")

        client.vector_stores.files.create(
            vector_store_id=VECTOR_STORE_ID,
            file_id=uploaded_file.id,
        )

        uploaded_files.append(
            {
                "filename": source_path.name,
                "file_id": uploaded_file.id,
            }
        )

        print("Added to vector store.")

    print()
    print("WAITING FOR PROCESSING")
    print("-" * 60)

    results: list[dict[str, str]] = []

    for uploaded in uploaded_files:
        status = wait_for_vector_store_file(
            client=client,
            vector_store_id=VECTOR_STORE_ID,
            file_id=uploaded["file_id"],
            filename=uploaded["filename"],
        )

        results.append(
            {
                **uploaded,
                "status": status,
            }
        )

    print()
    print("FINAL RESULTS")
    print("=" * 60)

    for result in results:
        print(
            f"{result['filename']}\n"
            f"  File ID: {result['file_id']}\n"
            f"  Status:  {result['status']}"
        )

    failed = [
        result
        for result in results
        if result["status"] != "completed"
    ]

    if failed:
        print()
        print(
            "One or more files did not finish successfully. "
            "Do not test retrieval until those errors are resolved."
        )
    else:
        print()
        print(
            "All three sources were successfully stored "
            "in the vector store."
        )


if __name__ == "__main__":
    main()
