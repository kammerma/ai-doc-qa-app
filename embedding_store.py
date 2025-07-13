import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_vector_store(text_chunks: list[str], collection_name: str = "docs"):
    chroma_client = chromadb.Client(Settings())
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for i, chunk in enumerate(text_chunks):
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        collection.add(documents=[chunk], embeddings=[embedding], ids=[str(i)])
    
    return collection
