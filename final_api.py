from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import os
import uuid

import uvicorn
from google.cloud import vision
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './cheating-app-460706-2c650aed4e38.json'
genai.configure(api_key=API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extract_mcq_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are an AI assistant that extracts questions from raw OCR text.
        Now extract from the following text:
    """),
    ("human", "{question_text}")
])

answer_mcq_template = ChatPromptTemplate.from_messages([
    ("system", """
You are a highly knowledgeable and precise AI tutor trained to assist learners with academic questions across all levels, from basic to highly advanced (e.g., IIT-level).

- You will receive a question that may or may not include answer choices (MCQ format).
- If answer choices are present, **analyze all options carefully** and return only the **full text of the correct option** — do not return the option letter (A, B, etc.), only the full content of the correct answer.
- If no options are given, provide a **concise yet complete answer** to the question.
- Focus on clarity, correctness, and educational value.
- Avoid unnecessary explanation unless the question explicitly asks for it.
- Do not fabricate options or assume missing ones.

Respond with only the correct answer.
"""),
    ("human", "{mcq_text}")
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=API_KEY)
parser = StrOutputParser()

# === Endpoint 1: Upload Image and OCR ===
@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=contents)
        response = client.text_detection(image=image)
        if response.error.message:
            raise HTTPException(status_code=500, detail=response.error.message)

        texts = response.text_annotations
        extracted_text = texts[0].description if texts else ""
        return {"ocr_text": extracted_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # === Endpoint 2: Extract MCQ from OCR Text ===
# @app.post("/extract_mcq/")
# async def extract_mcq(ocr_text: str = Form(...)):
#     try:
#         extract_chain = extract_mcq_template | llm | parser
#         extracted_mcq = extract_chain.invoke({"question_text": ocr_text})
#         return {"question": extracted_mcq}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# === Endpoint 3: Answer MCQ ===
@app.post("/answer_mcq/")
async def answer_mcq(mcq_text: str = Form(...)):
    try:
        answer_chain = answer_mcq_template | llm | parser
        final_answer = answer_chain.invoke({"mcq_text": mcq_text})
        return {"answer": final_answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
