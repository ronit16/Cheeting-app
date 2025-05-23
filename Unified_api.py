from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import os

import uvicorn
from google.cloud import vision
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# === Setup ===
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './cheating-app-460706-2c650aed4e38.json'

genai.configure(api_key=API_KEY)

app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Prompt Templates ===
extract_mcq_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are an AI assistant that extracts questions from raw OCR text.
        Now extract from the following text:
    """),
    ("human", "{question_text}")
])

answer_mcq_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are an AI assistant that answers questions.
        Just return the full text of the correct option.
    """),
    ("human", "{mcq_text}")
])

# === Model & Parser ===
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=API_KEY)
parser = StrOutputParser()

# === Unified Endpoint ===
@app.post("/process_ocr_question/")
async def process_ocr_and_answer(file: UploadFile = File(...)):
    try:
        # OCR Part
        contents = await file.read()
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=contents)
        response = client.text_detection(image=image)
        if response.error.message:
            raise HTTPException(status_code=500, detail=response.error.message)

        texts = response.text_annotations
        extracted_text = texts[0].description if texts else ""

        # MCQ Extraction
        extract_chain = extract_mcq_template | llm | parser
        extracted_mcq = extract_chain.invoke({"question_text": extracted_text})

        # MCQ Answering
        answer_chain = answer_mcq_template | llm | parser
        final_answer = answer_chain.invoke({"mcq_text": extracted_mcq})

        return {
            "ocr_text": extracted_text,
            "question": extracted_mcq,
            "answer": final_answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))