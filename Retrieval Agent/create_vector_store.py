from openai import OpenAI
from dotenv import load_dotenv
import os

print("Program started")

# Load your API key
load_dotenv()


client = OpenAI( #connects Python program to OpenAI
    api_key=os.getenv("OPENAI_API_KEY")
)


# Create a vector store

vector_store = client.vector_stores.create( #creates empty database
    name="Real ESSI Knowledge Base"
)


print(vector_store.id) #prints ID number of database
