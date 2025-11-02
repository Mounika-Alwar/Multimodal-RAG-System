from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import shutil
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.upload_pipeline import process_uploaded_file, save_vector_store
from backend.retrieval_pipeline import load_vector_store

# ------------------------
# Initialization
# ------------------------
load_dotenv()
app = FastAPI(title="Multimodal RAG System (Open-Source Embeddings + Gemini LLM)")

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

# ------------------------
# Gemini Model Loader
# ------------------------
def load_gemini():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.6,
        max_output_tokens=512,
        google_api_key=GOOGLE_API_KEY
    )

# ------------------------
# Upload Endpoint
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
# Query Endpoint
# ------------------------
@app.post("/query/")
async def query_documents(
    query: str = Form(...),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(4)
):
    vector_store = load_vector_store()
    if not vector_store:
        return JSONResponse({"error": "Vector store not found. Please upload files first."}, status_code=400)

    results_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    retrieved_docs = [doc for doc, _ in results_with_scores]

    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
You are a helpful AI assistant. Understand the context below to answer the question to your best.
If the context doesn’t at all contain atleast some traces of answer, say "I don't know."

Context:
{context_text}

Question: {query}
Answer:
"""

    llm = load_gemini()
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

    return JSONResponse({
        "query": query,
        "retrieved_documents": [
            {
                "content": doc.page_content,
                "metadata": getattr(doc, "metadata", {})
            } for doc, _ in results_with_scores
        ],
        "final_answer": answer.strip(),
        "model_used": "gemini-2.5-flash"
    })

@app.post("/clear_data/")
def clear_data():
    import shutil, os
    folders = ["backend/uploaded_files", "backend/vector_store"]
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    return {"status": "✅ Cleared uploaded_files and vector_store successfully"}


# ------------------------
# Root Endpoint
# ------------------------
@app.get("/")
def read_root():
    return {"status": "App is running successfully!"}

# ------------------------
# App Entry Point (for Render)
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
