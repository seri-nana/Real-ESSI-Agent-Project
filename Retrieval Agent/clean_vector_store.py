from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


VECTOR_STORE_ID = "vs_6a56bd35da488191a8efcdafc1e6272c"

FILES_TO_REMOVE = {
    "jeremic_lecture_notes_fr.pdf",
    "jeremic_lecture_notes_original.pdf",
}

FILES_TO_KEEP = {
    "jeremic_lecture_notes_verbatim.md",
    "jeremic_lecture_notes_figures.md",
}


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found in the .env file."
        )

    client = OpenAI(api_key=api_key)

    print("\nCURRENT VECTOR STORE FILES")
    print("=" * 70)

    response = client.vector_stores.files.list(
        vector_store_id=VECTOR_STORE_ID,
        limit=100,
    )

    vector_store_files = response.data

    if not vector_store_files:
        print("The vector store contains no files.")
        return

    files_removed = 0

    for vector_store_file in vector_store_files:
        file_id = vector_store_file.id

        file_object = client.files.retrieve(
            file_id=file_id
        )

        filename = file_object.filename

        print(f"\nFilename: {filename}")
        print(f"File ID:  {file_id}")
        print(f"Status:   {vector_store_file.status}")

        if filename in FILES_TO_REMOVE:
            print("Action:   Removing from vector store")

            client.vector_stores.files.delete(
                vector_store_id=VECTOR_STORE_ID,
                file_id=file_id,
            )

            files_removed += 1

        elif filename in FILES_TO_KEEP:
            print("Action:   Keeping")

        else:
            print("Action:   Leaving unchanged")

    print()
    print("=" * 70)
    print(f"Files removed from vector store: {files_removed}")

    print(
        "\nThe underlying OpenAI File objects were not deleted. "
        "Only their vector-store attachments were removed."
    )


if __name__ == "__main__":
    main()
