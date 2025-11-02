# Multimodal Retrieval-Augmented Generation (RAG) System

## Overview  
This project implements a **Multimodal Retrieval-Augmented Generation (RAG)** system capable of processing and querying **multiple data formats** — including text documents, images, and PDFs (text-based or mixed content).  
It builds a pipeline to extract meaningful content, convert it into embeddings, and perform semantic search via a vector database. The backend API enables ingestion, querying, and zero-effort reset of the memory store.

---

## Core Features  
- Supports uploads of plain text (`.txt`), images (`.png`, `.jpg`, `.jpeg`), and PDFs containing text and/or images.  
- Processes images into captions (using BLIP) for semantic representation alongside text.  
- Embeds content via a unified embedding model and stores in a **FAISS** vector store with metadata (file type, timestamp, source).  
- Enables cross-modal queries — for example, questions that reference image content and document content together.  
- Returns results with relevance scores and source metadata to ensure traceability.

---

## Architecture Overview

### 1. Data Ingestion & Processing  
- The upload endpoint accepts text, image and PDF files.  
- Text from PDFs is extracted using PyMuPDF.  
- Images are processed via a captioning model (BLIP) to generate descriptive text.  
- All output content (original or generated) is chunked and embedded.

### 2. Embedding & Storage  
- Text (from documents and captions) is embedded using a text-embedding model.  
- Embeddings are stored in FAISS with associated metadata such as document ID, page number and file type.  
- The vector store enables efficient similarity search.

### 3. Query Handling  
- The `/query` endpoint receives a user query (and optional image).  
- The system encodes the query, retrieves the top-k similar chunks from the vector store, and passes context to the chosen model (TinyLlama or Gemini).  
- A JSON response includes retrieved document content with scores and the final answer.

### 4. API Layer  
- Built with FastAPI.  
- Endpoints:  
  - `POST /upload` — Upload files and build embeddings.  
  - `POST /query` — Submit a query (with optional image) and receive an answer.  
  - `DELETE /reset-faiss` — Clear the vector store memory.  
  - `GET /` — Health/status endpoint.

---

## Branch Structure  

| Branch       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| **main**       | Primary implementation branch: full backend RAG system (text + images + PDFs). |
| **gemini-only** | Specialized branch for lighter deployment and integration with Google Gemini API.   |

---

## Tech Stack  

- **Backend Framework**: FastAPI  
- **Vector Database**: FAISS  
- **Embeddings & Captioning**: BLIP for images, text-embedding model for text  
- **Document Processing**: PyMuPDF for PDFs  
- **Language Models**: TinyLlama (local inference), Google Gemini (API)  
- **Containerization / Deployment**: Docker (attempted deployments on Render, DockerHub, cloud)  
- **Frontend (planned)**: Streamlit for interactive file upload and querying  

---

## Getting Started (Local Setup)  

### Prerequisites  
- Python 3.10 or higher  
- GPU support recommended for models  
- Ensure necessary packages (see `requirements.txt`)

### Installation  
```bash
git clone https://github.com/Mounika-Alwar/Multimodal-RAG-System.git
cd Multimodal-RAG-System
pip install -r requirements.txt

### To run FastAPI server
uvicorn backend.main:app --reload
