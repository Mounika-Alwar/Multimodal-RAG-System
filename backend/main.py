from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import shutil
import base64
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from backend.upload_pipeline import process_uploaded_file, save_vector_store, VECTOR_STORE_PATH

# ------------------------
# Initialization
# ------------------------
load_dotenv()
app = FastAPI(title="Multimodal RAG System (Gemini Only)")

# ------------------------
# Gemini Embedding Model
# ------------------------
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# ------------------------
# Vector Store Loader
# ------------------------
def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
    return None

# ------------------------
# Gemini Model Loader
# ------------------------
def load_gemini():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # ✅ latest stable multimodal model
        temperature=0.6,
        max_output_tokens=512,
        google_api_key=GOOGLE_API_KEY
    )

# ------------------------
# 1️⃣ Upload Endpoint
# ------------------------
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    temp_dir = "uploaded_files"
    os.makedirs(temp_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        process_uploaded_file(file_path)
        uploaded_files.append(file.filename)

    save_vector_store()
    return {"status": "success", "uploaded_files": uploaded_files}

# ------------------------
# 2️⃣ Query Endpoint (Gemini-only)
# ------------------------
@app.post("/query/")
async def query_documents(
    query: str = Form(...),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(4)
):
    # --- Load Vector Store ---
    vector_store = load_vector_store()
    if not vector_store:
        return JSONResponse({"error": "Vector store not found. Please upload files first."}, status_code=400)

    # --- Retrieve Relevant Docs ---
    results_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    retrieved_docs = [doc for doc, _ in results_with_scores]
    scores = [float(score) for _, score in results_with_scores]

    # --- Build Context ---
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
You are a helpful AI assistant that answers questions using the provided context.
If the context does not contain the answer, respond with "I don't know".

Context:
{context_text}

Question: {query}

Answer concisely and clearly:
"""

    # --- Load Gemini ---
    llm = load_gemini()

    # --- Prepare message (with optional image) ---
    message_content = [{"type": "text", "text": prompt}]
    if image:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        message_content.append({
            "type": "image_url",
            "image_url": f"data:{image.content_type};base64,{img_b64}"
        })

    messages = [{"role": "user", "content": message_content}]

    try:
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return JSONResponse({"error": f"Gemini generation failed: {str(e)}"}, status_code=500)

    # --- Build JSON Response ---
    return JSONResponse({
        "query": query,
        "model_used": "gemini-2.5-flash",
        "retrieved_documents": [
            {
                "content": doc.page_content,
                "score": float(score),
                "metadata": getattr(doc, "metadata", {})
            }
            for doc, score in results_with_scores
        ],
        "final_answer": answer.strip()
    })

# ------------------------
# Root Endpoint (for Render)
# ------------------------
@app.get("/")
def root():
    return {"message": "✅ Gemini-only Multimodal RAG API is running successfully on Render!"}

