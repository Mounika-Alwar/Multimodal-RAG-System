from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import shutil
import base64
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.upload_pipeline import process_uploaded_file, save_vector_store, VECTOR_STORE_PATH, embed_text

# ------------------------
# Initialization
# ------------------------
load_dotenv()
app = FastAPI(title="Multimodal RAG System")

# ------------------------
# Embedding Wrapper
# ------------------------
class CLIPTextEmbeddings:
    def embed_query(self, text): return embed_text(text)
    def embed_documents(self, texts): return [embed_text(t) for t in texts]
    def __call__(self, text): return embed_text(text)

clip_embedder = CLIPTextEmbeddings()

# ------------------------
# Load Vector Store
# ------------------------
if not os.path.exists(VECTOR_STORE_PATH):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

def load_vector_store():
    if os.listdir(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings=clip_embedder, allow_dangerous_deserialization=True)
    else:
        return None

# ------------------------
# Gemini Model Loader
# ------------------------
def load_gemini():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # ✅ latest stable model
        temperature=0.6,
        max_output_tokens=512,
        google_api_key=os.getenv("GEMINI_API_KEY")
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

    # --- Prepare Combined Context ---
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
You are a helpful assistant that answers based on the context below by understanding the context in the detailed way.
If the context does not at all contain anything related to question, reply "I don't know".

Context:
{context_text}

Question: {query}

Answer concisely and clearly:
"""

    # --- Load Gemini ---
    llm = load_gemini()

    # --- Handle optional image input ---
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
    return {"message": "API is running successfully on Render with Gemini!"}

