from openai import OpenAI
from dotenv import load_dotenv
import os


VECTOR_STORE_ID = "vs_6a56bd35da488191a8efcdafc1e6272c"
FAILED_FILE_ID = "file-RwgshiJuFW776Xw8gpAH8j"


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

client.vector_stores.files.delete(
    vector_store_id=VECTOR_STORE_ID,
    file_id=FAILED_FILE_ID,
)

print("Failed file removed from the vector store.")
