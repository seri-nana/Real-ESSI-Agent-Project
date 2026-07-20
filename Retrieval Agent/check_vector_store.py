from openai import OpenAI
from dotenv import load_dotenv
import os


VECTOR_STORE_ID = "vs_6a56bd35da488191a8efcdafc1e6272c"


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY was not found in the .env file.")

client = OpenAI(api_key=api_key)


print("\nCHECKING VECTOR STORE")
print("-" * 50)

# Retrieve information about the vector store
vector_store = client.vector_stores.retrieve(
    vector_store_id=VECTOR_STORE_ID
)

print("Vector store ID:", vector_store.id)
print("Vector store name:", vector_store.name)
print("Vector store status:", vector_store.status)
print("File counts:", vector_store.file_counts)
print("Usage bytes:", vector_store.usage_bytes)


print("\nFILES ATTACHED TO VECTOR STORE")
print("-" * 50)

files = client.vector_stores.files.list(
    vector_store_id=VECTOR_STORE_ID
)

if not files.data:
    print("No files are attached to this vector store.")
else:
    for file in files.data:
        print("File ID:", file.id)
        print("Status:", file.status)
        print("Usage bytes:", file.usage_bytes)

        if getattr(file, "last_error", None):
            print("Last error:", file.last_error)

        print("-" * 30)


print("\nTESTING A BASIC SEARCH")
print("-" * 50)

results = client.vector_stores.search(
    vector_store_id=VECTOR_STORE_ID,
    query="domain reduction method",
    max_num_results=10,
)

print("Number of search results:", len(results.data))

for number, result in enumerate(results.data, start=1):
    print(f"\nRESULT {number}")
    print("Filename:", result.filename)
    print("Score:", result.score)

    for content in result.content:
        if content.type == "text":
            print("Text:")
            print(content.text[:1000])
