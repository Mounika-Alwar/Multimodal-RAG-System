import os
import io
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ------------------------------------------------------
# Load environment
# ------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

# ------------------------------------------------------
# Initialize Gemini Embedding Model
# ------------------------------------------------------
print("🚀 Loading Gemini Embedding Model...")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
print("✅ Gemini Embedding Model Ready!\n")

# ------------------------------------------------------
# Globals
# ------------------------------------------------------
VECTOR_STORE_PATH = "vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_docs = []
all_embeddings = []

# ------------------------------------------------------
# Text Processing
# ------------------------------------------------------
def process_text_file(file_path):
    print(f"📝 Processing text file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    temp_doc = Document(page_content=text, metadata={"type": "text", "source": file_path})
    chunks = splitter.split_documents([temp_doc])
    all_docs.extend(chunks)

# ------------------------------------------------------
# PDF Processing
# ------------------------------------------------------
def process_pdf_file(file_path):
    print(f"📄 Processing PDF file: {file_path}")
    pdf_doc = fitz.open(file_path)

    for page_idx, page in enumerate(pdf_doc):
        text = page.get_text("text").strip()
        if text:
            temp_doc = Document(
                page_content=text,
                metadata={"type": "pdf_text", "page": page_idx, "source": file_path}
            )
            chunks = splitter.split_documents([temp_doc])
            all_docs.extend(chunks)

        # Process images (optional, store image path metadata only)
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_path = os.path.join(VECTOR_STORE_PATH, f"pdf_page{page_idx}_img{img_index}.png")

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                img_doc = Document(
                    page_content=f"[Image from {os.path.basename(file_path)} page {page_idx}]",
                    metadata={
                        "type": "image",
                        "source": file_path,
                        "page": page_idx,
                        "image_path": image_path
                    },
                )
                all_docs.append(img_doc)
            except Exception as e:
                print(f"⚠️ Error extracting image {img_index} from page {page_idx}: {e}")

# ------------------------------------------------------
# Image Processing
# ------------------------------------------------------
def process_image_file(file_path):
    print(f"🖼️ Processing image file: {file_path}")
    # No caption generation needed; just store metadata for reference
    doc = Document(
        page_content=f"[Image: {os.path.basename(file_path)}]",
        metadata={"type": "image", "source": file_path, "image_path": file_path}
    )
    all_docs.append(doc)

# ------------------------------------------------------
# Dispatcher
# ------------------------------------------------------
def process_uploaded_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        process_text_file(file_path)
    elif ext == ".pdf":
        process_pdf_file(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        process_image_file(file_path)
    else:
        print(f"❌ Unsupported file type: {file_path}")

# ------------------------------------------------------
# Save Vector Store
# ------------------------------------------------------
def save_vector_store():
    print(f"\n💾 Embedding {len(all_docs)} documents using Gemini Embeddings...")
    vector_store = FAISS.from_documents(all_docs, embedding_model)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ Vector store saved at: {VECTOR_STORE_PATH}")

# ------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------
if __name__ == "__main__":
    print("📂 Gemini Upload Pipeline Ready!\n")

    folder_path = input("📁 Enter folder path to process: ").strip()

    if not os.path.exists(folder_path):
        raise ValueError("❌ Folder not found!")

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        process_uploaded_file(file_path)

    save_vector_store()
    print("\n✅ Upload pipeline completed successfully!")

