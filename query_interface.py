import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_vector_store(question: str, collection, top_k: int = 3):
    response = client.embeddings.create(
        input=question,
        model="text-embedding-ada-002"
    )
    question_embedding = response.data[0].embedding
    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
    return results["documents"][0]
