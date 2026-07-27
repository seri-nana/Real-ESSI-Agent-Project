from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


VECTOR_STORE_ID = "vs_6a56bd35da488191a8efcdafc1e6272c"

TEST_QUERIES = [
    "domain reduction method",
    "force equilibrium in Cosserat continua",
    "moment equilibrium in Cosserat continua",
]


def get_result_text(result) -> str:
    """Combine all text portions returned for one search result."""
    text_parts: list[str] = []

    for content_item in result.content:
        text = getattr(content_item, "text", None)

        if text:
            text_parts.append(text)

    return "\n".join(text_parts).strip()


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found in the .env file."
        )

    client = OpenAI(api_key=api_key)

    for query in TEST_QUERIES:
        print()
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = client.vector_stores.search(
            vector_store_id=VECTOR_STORE_ID,
            query=query,
            max_num_results=10,
        )

        if not results.data:
            print("No results were returned.")
            continue

        for result_number, result in enumerate(
            results.data,
            start=1,
        ):
            result_text = get_result_text(result)

            print()
            print(f"RESULT {result_number}")
            print("-" * 80)
            print(f"Filename: {result.filename}")
            print(f"File ID:  {result.file_id}")
            print(f"Score:    {result.score:.4f}")
            print()
            print(result_text[:2500])

            if len(result_text) > 2500:
                print("\n[Result shortened for terminal display]")


if __name__ == "__main__":
    main()
