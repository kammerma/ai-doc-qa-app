import openai
import chromadb
from chromadb.config import Settings

def create_vector_store(text_chunks: list[str], collection_name: str = "docs"):
    client = chromadb.Client(Settings())
    collection = client.get_or_create_collection(name=collection_name)

    for i, chunk in enumerate(text_chunks):
        embedding = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")["data"][0]["embedding"]
        collection.add(documents=[chunk], embeddings=[embedding], ids=[str(i)])
    
    return collection
