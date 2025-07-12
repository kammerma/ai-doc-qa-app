import openai

def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n".join(context_chunks)
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]
