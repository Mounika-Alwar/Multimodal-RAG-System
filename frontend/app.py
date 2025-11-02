import streamlit as st
import requests
import base64

# -------------------------------
# Configuration
# -------------------------------
API_BASE_URL = "http://127.0.0.1:8000/"  # change if backend is hosted remotely

st.set_page_config(page_title="Multimodal RAG System", layout="wide")

# -------------------------------
# Title and Description
# -------------------------------
st.title("🧠 Multimodal RAG System")
st.markdown("""
This application allows you to **upload text and image files**, build embeddings with CLIP,
and query them using **TinyLlama** or **Gemini** models.

### 🔍 Unique Features
1. **Multimodal Querying** — Upload both text and image data for context retrieval.  
2. **FAISS Vector Store** — Efficient similarity search and memory persistence.  
3. **LLM Choice** — Select between open-source `TinyLlama` and Google’s `Gemini`.  
4. **Top-K Retrieval** — Control how many relevant chunks are fetched.  
5. **Interactive UI** — Real-time document upload, query, and visual results display.
""")

st.divider()

# -------------------------------
# Sidebar for LLM and Parameters
# -------------------------------
st.sidebar.header("⚙️ Configuration")

llm_choice = st.sidebar.radio("Choose Model", ("tinyllama", "gemini"))
top_k = st.sidebar.slider("Number of Retrieved Documents (k)", min_value=1, max_value=10, value=4)

# -------------------------------
# Upload Section
# -------------------------------
st.header("📁 Upload Documents")
uploaded_files = st.file_uploader("Upload one or more files", type=["txt", "pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("Upload to Server"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        with st.spinner("Uploading and processing files..."):
            files_payload = [("files", (f.name, f, f.type)) for f in uploaded_files]
            response = requests.post(f"{API_BASE_URL}/upload/", files=files_payload)
        if response.status_code == 200:
            st.success("✅ Files uploaded and embedded successfully!")
            st.json(response.json())
        else:
            st.error(f"❌ Upload failed: {response.text}")

st.divider()

# -------------------------------
# Query Section
# -------------------------------
st.header("💬 Query the RAG System")
query = st.text_area("Enter your query", placeholder="Ask something about the uploaded documents...")
image_file = st.file_uploader("Optionally upload an image to query with", type=["png", "jpg", "jpeg"])

if st.button("Submit Query"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing your query..."):
            data = {"query": query, "model_choice": llm_choice, "top_k": top_k}
            files = {}
            if image_file:
                files = {"image": (image_file.name, image_file, image_file.type)}

            try:
                response = requests.post(f"{API_BASE_URL}/query/", data=data, files=files)
                if response.status_code == 200:
                    result = response.json()

                    st.subheader("📄 Retrieved Documents")
                    for i, doc in enumerate(result["retrieved_documents"], 1):
                        with st.expander(f"Document {i} (Score: {doc['score']:.4f})"):
                            st.write(doc["content"])
                            if doc.get("metadata"):
                                st.json(doc["metadata"])

                    st.divider()
                    st.subheader("🧠 Final Answer")
                    st.markdown(f"**{result['final_answer']}**")

                else:
                    st.error(f"❌ Query failed: {response.text}")
            except Exception as e:
                st.error(f"⚠️ Error connecting to backend: {e}")

st.divider()

# -------------------------------
# Reset FAISS Memory
# -------------------------------
st.header("🧹 Reset FAISS Memory")
if st.button("Clear All Stored Embeddings"):
    with st.spinner("Clearing FAISS vector store..."):
        resp = requests.delete(f"{API_BASE_URL}/reset-faiss/")
        if resp.status_code == 200:
            st.success("✅ FAISS memory cleared successfully.")
        else:
            st.error(f"❌ Failed to clear FAISS: {resp.text}")
