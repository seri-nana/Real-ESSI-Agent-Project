from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import sys


# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_STORE_ID = "vs_6a56bd35da488191a8efcdafc1e6272c"

PDF_PATH = Path(
    "documents/jeremic_lecture_notes_fr.pdf"
)


# ============================================================
# CONNECT TO OPENAI
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: OPENAI_API_KEY was not found in the .env file.")
    sys.exit(1)

client = OpenAI(api_key=api_key)


# ============================================================
# CHECK THAT THE PDF EXISTS
# ============================================================

if not PDF_PATH.exists():
    print("\nERROR: The repaired PDF could not be found.")
    print("Expected location:")
    print(PDF_PATH.resolve())
    sys.exit(1)


# ============================================================
# UPLOAD THE REPAIRED PDF
# ============================================================

print("\nUploading repaired PDF...")

try:
    with PDF_PATH.open("rb") as pdf_file:
        uploaded_file = client.files.create(
            file=pdf_file,
            purpose="assistants",
        )
except Exception as error:
    print("\nThe PDF upload failed.")
    print("Error details:", error)
    sys.exit(1)

print("File uploaded successfully.")
print("OpenAI file ID:", uploaded_file.id)


# ============================================================
# ADD THE PDF TO THE VECTOR STORE
# ============================================================

print("\nAdding the PDF to the vector store...")
print("Waiting for OpenAI to parse and index the file...")

try:
    vector_store_file = client.vector_stores.files.create_and_poll(
        vector_store_id=VECTOR_STORE_ID,
        file_id=uploaded_file.id,
    )
except Exception as error:
    print("\nThe file could not be added to the vector store.")
    print("Error details:", error)
    sys.exit(1)


# ============================================================
# DISPLAY THE RESULT
# ============================================================

print("\nIndexing finished.")
print("Vector-store file ID:", vector_store_file.id)
print("Status:", vector_store_file.status)
print("Usage bytes:", vector_store_file.usage_bytes)

if vector_store_file.status != "completed":
    print("\nThe repaired PDF still could not be indexed.")

    if getattr(vector_store_file, "last_error", None):
        print("OpenAI error:", vector_store_file.last_error)

    sys.exit(1)

print("\nSUCCESS: The repaired PDF is ready to search.")

# context.txt
