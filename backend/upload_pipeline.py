import os
import pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_STORE_PATH = "vector_store"

# Use open-source sentence-transformer embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_uploaded_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt"]:
        loader = TextLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        print(f"⚠️ Skipping unsupported file type: {ext}")
        return None

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create or update FAISS store
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        vector_store.add_documents(docs)
    else:
        vector_store = FAISS.from_documents(docs, embedding_model)

    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ Processed and stored embeddings for: {file_path}")

def save_vector_store():
    print("✅ Vector store saved successfully.")
