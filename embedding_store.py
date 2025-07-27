import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv


load_dotenv()

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)


def create_vector_store(
    text_chunks: list[str],
    collection_name: str = "docs",
    embedding_model: str = None,
    batch_size: int = 10
) -> chromadb.api.models.Collection:
    """
    Creates a ChromaDB collection with embeddings for the provided text chunks.
    Args:
        text_chunks: List of text strings to embed and store.
        collection_name: Name of the ChromaDB collection.
        embedding_model: OpenAI embedding model to use (default: text-embedding-ada-002 or from env).
        batch_size: Number of chunks to process per API call (if supported).
    Returns:
        The ChromaDB collection with added embeddings.
    Raises:
        ValueError: If OpenAI API key is not set.
        Exception: For API or DB errors.
    """
    chroma_client = chromadb.Client(Settings())
    collection = chroma_client.get_or_create_collection(name=collection_name)
    client = get_openai_client()
    model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

    try:
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            embeddings = [item.embedding for item in response.data]
            ids = [str(j) for j in range(i, i+len(batch))]
            collection.add(documents=batch, embeddings=embeddings, ids=ids)
    except Exception as e:
        print(f"Error creating embeddings or adding to collection: {e}")
        raise

    return collection
