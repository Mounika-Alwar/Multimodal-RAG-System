import os
import base64
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from upload_pipeline import embed_text

# Load environment variables
load_dotenv()

# -------------------------------
# Embedding Wrapper
# -------------------------------
class CLIPTextEmbeddings:
    def embed_query(self, text):
        return embed_text(text)
    def embed_documents(self, texts):
        return [embed_text(t) for t in texts]
    def __call__(self, text):
        return embed_text(text)

clip_embedder = CLIPTextEmbeddings()

# -------------------------------
# Load Vector Store
# -------------------------------
VECTOR_STORE_PATH = "vector_store"
if not os.path.exists(VECTOR_STORE_PATH):
    raise ValueError("❌ Vector store not found. Please run upload_pipeline.py first!")

print("📦 Loading FAISS vector store...")
vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings=clip_embedder, allow_dangerous_deserialization=True)
print("✅ Vector store loaded successfully!\n")

# -------------------------------
# Choose Model
# -------------------------------
print("🤖 Choose the model you wish to use:")
print("1️⃣  TinyLlama (for text-based Q&A)")
print("2️⃣  Gemini 1.5 Flash (for multimodal queries)")

choice = input("Enter model number: ").strip()

# -------------------------------
# Model 1: TinyLlama
# -------------------------------
if choice == "1":
    print("\n🚀 Loading TinyLlama model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=300,
        temperature=0.7,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ TinyLlama loaded successfully!\n")

# -------------------------------
# Model 2: Gemini (Multimodal)
# -------------------------------
elif choice == "2":
    print("\n🚀 Loading Gemini 2.5 Flash (multimodal)...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # ✅ Correct latest supported model
        temperature=0.6,
        max_output_tokens=512,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    print("✅ Gemini ready!\n")

else:
    print("❌ Invalid choice. Exiting...")
    exit()

# -------------------------------
# Query Loop
# -------------------------------
while True:
    query = input("\n🔍 Enter your query (or type 'exit' to quit): ").strip()
    if query.lower() == "exit":
        print("👋 Exiting retrieval pipeline.")
        break

    # -------------------------------
    # Optional Image Input
    # -------------------------------
    image_path = input("🖼️  (Optional) Enter image path or press Enter to skip: ").strip()
    retrieved_image_paths = []
    retrieved_text_chunks = []

    print("🧠 Searching for relevant chunks...")
    results = vector_store.similarity_search(query, k=4)
    retrieved_text_chunks = [doc.page_content for doc in results]

    # Collect any images from metadata
    for doc in results:
        if "image_path" in doc.metadata:
            retrieved_image_paths.append(doc.metadata["image_path"])

    print(f"📄 Found {len(retrieved_text_chunks)} text chunks and {len(retrieved_image_paths)} images.")

    # -------------------------------
    # TinyLlama Response
    # -------------------------------
    if choice == "1":
        context = "\n\n".join(retrieved_text_chunks)
        prompt = f"""
        You are a helpful assistant that answers ONLY the user's question.
        Ignore any 'Question:' or 'Answer:' text in the context — they are just examples.

        Context:
        {context}

        User's question: {query}

        Your answer (concise, clear, and single paragraph):
        """
        print("💬 Generating response...\n")
        output = llm.invoke(prompt)
        print("🧩 Answer:", output, "\n" + "-" * 80)

    # -------------------------------
    # Gemini Multimodal Response
    # -------------------------------
    elif choice == "2":
        print("\n🚀 Loading Gemini 2.5 Flash (multimodal)...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.6,
            max_output_tokens=512,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        print("✅ Gemini ready!\n")

    else:
        print("❌ Invalid choice. Exiting...")
        exit()

    # -------------------------------
    # Query Function
    # -------------------------------
    def retrieval_query(query, image_path=None):
        print("🧠 Searching for relevant chunks...")
        results = vector_store.similarity_search(query, k=4)
        retrieved_text_chunks = [doc.page_content for doc in results]

        # Extract any linked images from metadata
        retrieved_image_paths = [
            doc.metadata.get("image_path") for doc in results
            if isinstance(doc.metadata, dict) and "image_path" in doc.metadata
        ]

        print(f"📄 Found {len(retrieved_text_chunks)} text chunks and {len(retrieved_image_paths)} images.")
        return retrieved_text_chunks, retrieved_image_paths

    # -------------------------------
    # Multimodal Query Builder (Gemini)
    # -------------------------------
    def build_gemini_message(query, retrieved_text_chunks, all_image_paths):
        # Construct text portion
        context_text = "\n\n".join(retrieved_text_chunks)
        base_content = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "You are a helpful assistant that answers questions using the given context. "
                        "If the context does not contain the answer, respond with 'I don't know'.\n\n"
                        f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer clearly and concisely:"
                    )}
                ]
            }
        ]

        # Add images if present
        for img_path in all_image_paths:
            try:
                with open(img_path, "rb") as img_file:
                    img_bytes = img_file.read()
                base_content[0]["content"].append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"
                })
            except Exception as e:
                print(f"⚠️ Could not attach image {img_path}: {e}")

        return base_content

    # -------------------------------
    # Query Loop
    # -------------------------------
    while True:
        query = input("\n🔍 Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("👋 Exiting retrieval pipeline.")
            break

        image_path = input("🖼️  (Optional) Enter image path or press Enter to skip: ").strip()
        retrieved_text_chunks, retrieved_image_paths = retrieval_query(query, image_path)

        # -------------------------------
        # Text-Only Model
        # -------------------------------
        if choice == "1":
            context = "\n\n".join(retrieved_text_chunks)
            prompt = f"""
            You are a helpful assistant that answers ONLY the user's question.
            Ignore any 'Question:' or 'Answer:' text in the context — they are just examples.

            Context:
            {context}

            User's question: {query}

            Your answer (concise, clear, and single paragraph):
            """
            print("💬 Generating response...\n")
            output = llm(prompt)
            print("🧩 Answer:", output, "\n" + "-" * 80)

        # -------------------------------
        # Multimodal Model (Gemini)
        # -------------------------------
        elif choice == "2":
            all_image_paths = []
            if image_path and os.path.exists(image_path):
                all_image_paths.append(image_path)
            all_image_paths.extend(retrieved_image_paths)

            messages = build_gemini_message(query, retrieved_text_chunks, all_image_paths)

            try:
                print(f"🖼️ Sending {len(all_image_paths)} images and {len(retrieved_text_chunks)} text chunks to Gemini...")
                response = llm.invoke(messages)
                print("\n🤖 Gemini Response:")
                print(response.content)
            except Exception as e:
                print(f"❌ Error while calling Gemini: {e}")

            print("-" * 80)