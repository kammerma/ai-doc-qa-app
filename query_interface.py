import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_vector_store(question: str, collection, top_k: int = 3):
    question_embedding = openai.Embedding.create(input=question, model="text-embedding-ada-002")["data"][0]["embedding"]
    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
    return results["documents"][0]
