from fastapi import FastAPI, UploadFile, Form
from input_handler import save_uploaded_file
from ocr_azure import extract_text_from_pdf
from embedding_store import create_vector_store
from query_interface import query_vector_store
from llm_response import generate_answer

app = FastAPI()
collection = None

@app.post("/upload/")
async def upload_pdf(file: UploadFile):
    file_path = save_uploaded_file(file)
    text = extract_text_from_pdf(file_path, endpoint="YOUR_AZURE_ENDPOINT", key="YOUR_AZURE_KEY")
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    global collection
    collection = create_vector_store(chunks)
    return {"message": "Document processed and indexed."}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    if not collection:
        return {"error": "No document uploaded yet."}
    context = query_vector_store(question, collection)
    answer = generate_answer(question, context)
    return {"answer": answer}
