from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(file_path: str) -> str:
    endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-read", document=f)
    result = poller.result()

    return "\n".join([line.content for page in result.pages for line in page.lines])
