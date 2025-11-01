import os
import base64
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ------------------------------------------------------
# Load environment variables
# ------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Please set GEMINI_API_KEY in your .env file")

# ------------------------------------------------------
# Initialize Gemini Embeddings
# ------------------------------------------------------
print("🚀 Initializing Gemini Embeddings...")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
print("✅ Gemini Embeddings ready!\n")

# ------------------------------------------------------
# Load Vector Store
# ------------------------------------------------------
VECTOR_STORE_PATH = "vector_store"

if not os.path.exists(VECTOR_STORE_PATH):
    raise ValueError("❌ Vector store not found! Please run upload_pipeline.py first to build it.")

print("📦 Loading FAISS vector store...")
vector_store = FAISS.load_local(
    VECTOR_STORE_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
print("✅ Vector store loaded successfully!\n")

# ------------------------------------------------------
# Initialize Gemini LLM
# ------------------------------------------------------
print("🚀 Loading Gemini 2.5 Flash (multimodal)...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.6,
    max_output_tokens=512,
    google_api_key=GOOGLE_API_KEY
)
print("✅ Gemini LLM ready!\n")

# ------------------------------------------------------
# Function: Retrieve relevant chunks
# ------------------------------------------------------
def retrieve_chunks(query):
    """Search FAISS for most relevant text/image chunks"""
    print("🔎 Retrieving relevant chunks...")
    results = vector_store.similarity_search(query, k=4)

    retrieved_texts = [doc.page_content for doc in results]
    retrieved_images = [
        doc.metadata.get("image_path")
        for doc in results
        if isinstance(doc.metadata, dict) and "image_path" in doc.metadata
    ]

    print(f"📄 Retrieved {len(retrieved_texts)} text chunks, 🖼️ {len(retrieved_images)} image references.\n")
    return retrieved_texts, retrieved_images

# ------------------------------------------------------
# Function: Build Gemini multimodal message
# ------------------------------------------------------
def build_message(query, retrieved_texts, image_paths):
    """Prepare messages for Gemini model"""
    context = "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."

    content = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a helpful AI assistant. Use the context below to answer the question.\n\n"
                        "If the context doesn't have enough information, say 'I don’t know'.\n\n"
                        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                    ),
                }
            ],
        }
    ]

    # Attach images as base64 (if any)
    for img_path in image_paths:
        if img_path and os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                content[0]["content"].append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{img_data}"
                })
            except Exception as e:
                print(f"⚠️ Could not attach image {img_path}: {e}")

    return content

# ------------------------------------------------------
# Main Query Loop
# ------------------------------------------------------
if __name__ == "__main__":
    print("🤖 Gemini Retrieval-Augmented QA System Ready!\n")

    while True:
        query = input("💬 Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("👋 Exiting Gemini Retrieval Pipeline.")
            break

        image_input = input("🖼️  (Optional) Enter image path or press Enter to skip: ").strip()

        # Retrieve context
        retrieved_texts, retrieved_images = retrieve_chunks(query)

        # Combine input image + retrieved images
        all_images = []
        if image_input and os.path.exists(image_input):
            all_images.append(image_input)
        all_images.extend(retrieved_images)

        # Build and send to Gemini
        messages = build_message(query, retrieved_texts, all_images)

        try:
            print("🤔 Generating response from Gemini...\n")
            response = llm.invoke(messages)
            print("🧠 Gemini Response:\n")
            print(response.content)
        except Exception as e:
            print(f"❌ Error during Gemini call: {e}")

        print("\n" + "-" * 80 + "\n")
