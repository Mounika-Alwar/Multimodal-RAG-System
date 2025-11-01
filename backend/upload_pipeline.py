# upload_pipeline.py
import os
import io
import base64
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitters import RecursiveCharacterTextSplitter
import torch

# Load environment
load_dotenv()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize lightweight BLIP captioning model (image -> short text)
# Using a relatively small BLIP checkpoint for captions
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

blip_model.eval()
clip_model.eval()

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Storage
all_docs = []
all_embeddings = []

# Output folder
VECTOR_STORE_PATH = "vector_store"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)


# --- Embedding Functions ---
def embed_text(text: str):
    """Embed text using CLIP text encoder."""
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().numpy()


def generate_image_caption(image_data):
    """Generate textual description (caption) for an image using BLIP."""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data

    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()


# --- File Processing Functions ---
def process_text_file(file_path):
    print(f"📝 Processing text file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    temp_doc = Document(page_content=text, metadata={"type": "text", "source": file_path})
    chunks = splitter.split_documents([temp_doc])

    for chunk in chunks:
        emb = embed_text(chunk.page_content)
        all_docs.append(chunk)
        all_embeddings.append(emb)


def process_image_file(file_path):
    print(f"🖼️ Processing image file: {file_path}")
    caption = generate_image_caption(file_path)
    emb = embed_text(caption)
    doc = Document(
        page_content=f"[Image: {os.path.basename(file_path)}] {caption}",
        metadata={"type": "image", "source": file_path, "caption": caption}
    )
    all_docs.append(doc)
    all_embeddings.append(emb)


def process_pdf_file(file_path):
    print(f"📄 Processing PDF file: {file_path}")
    doc = fitz.open(file_path)

    for page_idx, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"type": "text", "page": page_idx, "source": file_path})
            chunks = splitter.split_documents([temp_doc])
            for chunk in chunks:
                emb = embed_text(chunk.page_content)
                all_docs.append(chunk)
                all_embeddings.append(emb)

        # Process images
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                caption = generate_image_caption(pil_image)
                emb = embed_text(caption)

                img_doc = Document(
                    page_content=f"[Image in {os.path.basename(file_path)} page {page_idx}] {caption}",
                    metadata={
                        "type": "image",
                        "source": file_path,
                        "page": page_idx,
                        "caption": caption
                    },
                )
                all_docs.append(img_doc)
                all_embeddings.append(emb)
            except Exception as e:
                print(f"⚠️ Error processing image {img_index} on page {page_idx}: {e}")
                continue

def process_uploaded_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt"]:
        process_text_file(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        process_image_file(file_path)
    elif ext == ".pdf":
        process_pdf_file(file_path)
    else:
        print(f"❌ Unsupported file type: {file_path}")


# --- Save FAISS index ---
def save_vector_store():
    print(f"\n💾 Saving FAISS index with {len(all_docs)} documents...")
    embeddings_array = np.array(all_embeddings)
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
        embedding=None,
        metadatas=[doc.metadata for doc in all_docs],
    )
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ Saved vector store to {VECTOR_STORE_PATH}")
