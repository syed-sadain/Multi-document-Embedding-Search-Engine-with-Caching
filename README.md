# 🔎 Multi-Document Embedding Search Engine

> **AI-powered semantic search engine for finding relevant information across multiple documents using transformer-based embeddings, cosine similarity, and an intelligent caching layer.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/NLP-Embeddings-orange)]()
[![Semantic Search](https://img.shields.io/badge/Semantic-Search-purple)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 📌 Overview

**Multi-Document Embedding Search Engine** is an AI-powered semantic search application that retrieves the most relevant information from multiple documents based on **meaning rather than exact keyword matching**.

The system converts document content and user queries into numerical **vector embeddings** and compares them using **cosine similarity** to identify the most relevant text chunks.

To improve performance, the application includes a **caching mechanism** that stores previously generated embeddings and metadata, preventing unnecessary recomputation.

### 🎯 Key Highlights

* 📚 Multi-document ingestion
* ✂️ Intelligent text chunking
* 🧠 Transformer-based embeddings
* 🔎 Semantic similarity search
* ⚡ Embedding caching
* 🚀 Fast retrieval
* 📊 Similarity-based ranking
* 🔌 FastAPI backend
* 🖥️ Streamlit web interface
* 📄 Support for large documents

---

## ✨ Features

### 📚 Multi-Document Processing

Upload and process multiple documents within the same search system.

### 🧠 Transformer-Based Embeddings

Documents are converted into meaningful vector representations using modern embedding models such as:

* Sentence Transformers
* OpenAI Embeddings

### 🔎 Semantic Search

Instead of relying only on keyword matching, the system understands the **semantic meaning** of the query and retrieves relevant content.

### ⚡ Intelligent Caching

Generated embeddings are stored locally so the application does not need to regenerate embeddings for unchanged documents.

This significantly reduces:

* Processing time
* API/model calls
* Computational overhead
* Repeated embedding generation

### ✂️ Text Chunking

Large documents are divided into smaller chunks before generating embeddings, allowing the system to efficiently process large amounts of text.

### 📊 Similarity Ranking

Search results are ranked according to their cosine similarity score, allowing the most relevant chunks to appear first.

### 🔌 REST API

The backend provides API endpoints through **FastAPI**, making the search engine accessible programmatically.

### 🖥️ Streamlit Interface

A simple and interactive Streamlit interface allows users to upload documents and perform semantic searches without directly interacting with the API.

---

## 🧠 How It Works

The application follows the following pipeline:

```text
                📄 Documents
                     │
                     ▼
            Document Preprocessing
                     │
                     ▼
               Text Chunking
                     │
                     ▼
          🧠 Embedding Generation
                     │
                     ▼
              💾 Cache Storage
                     │
                     ▼
              🔎 Search Index
                     │
                     │
User Query ──────────┤
                     ▼
             Query Embedding
                     │
                     ▼
          Cosine Similarity Search
                     │
                     ▼
          📊 Similarity Ranking
                     │
                     ▼
            🎯 Top Relevant Results
```

### Search Process

1. 📥 Load multiple documents.
2. 🧹 Preprocess the document content.
3. ✂️ Split documents into manageable text chunks.
4. 🧠 Generate embeddings for each chunk.
5. 💾 Store embeddings and metadata in the cache.
6. 🔎 Accept a user search query.
7. 🧠 Generate an embedding for the query.
8. 📐 Calculate cosine similarity between the query and document embeddings.
9. 📊 Rank results according to similarity.
10. 🎯 Return the most relevant document chunks.

---

## ⚡ Caching Architecture

One of the main performance improvements in this project is the embedding cache.

Without caching:

```text
Document
   ↓
Generate Embedding
   ↓
Generate Embedding Again
   ↓
Generate Embedding Again
```

With caching:

```text
Document
   ↓
Check Cache
   │
   ├── ✅ Embedding Exists → Reuse
   │
   └── ❌ Not Found → Generate → Store → Reuse
```

### Cache Storage

The project uses local cache files such as:

```text
cache/
├── index_meta.pkl
├── embeddings_cache.db
└── documents.index
```

These files store information required to avoid unnecessary embedding generation and speed up subsequent searches.

---

## 🏗️ Project Structure

```text
project/
│
├── src/
│   └── ...
│
├── data/
│   └── ...
│
├── cache/
│   ├── index_meta.pkl
│   ├── embeddings_cache.db
│   └── documents.index
│
├── appx.py
├── ui.py
├── requirements.txt
├── .gitignore
└── README.md
```

### 📂 Important Files

| File / Directory   | Purpose                              |
| ------------------ | ------------------------------------ |
| `appx.py`          | Backend/API application              |
| `ui.py`            | Streamlit user interface             |
| `src/`             | Core application/source code         |
| `data/`            | Input documents and application data |
| `cache/`           | Cached embeddings and metadata       |
| `requirements.txt` | Python dependencies                  |
| `.gitignore`       | Ignored files and directories        |
| `README.md`        | Project documentation                |

---

## 🛠️ Tech Stack

### 👨‍💻 Programming

* Python

### 🧠 AI / NLP

* Sentence Transformers
* OpenAI Embeddings
* NLP Preprocessing
* Transformer Models

### 🔎 Search & Similarity

* Semantic Search
* Cosine Similarity
* Vector Embeddings
* Similarity Ranking

### ⚡ Backend

* FastAPI

### 🖥️ Frontend / UI

* Streamlit

### 💾 Storage & Caching

* SQLite
* Pickle
* Local Embedding Cache

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <PROJECT_DIRECTORY>
```

### 2️⃣ Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### 🚀 Start the Backend

Open a terminal and run:

```bash
python appx.py
```

The backend server should remain running while using the application.

---

### 🖥️ Start the Streamlit UI

Open another terminal:

```bash
streamlit run ui.py
```

Streamlit will provide a local URL where you can access the application.

---

## 🔌 FastAPI Documentation

Once the backend is running, open:

```text
http://127.0.0.1:8000/docs
```

FastAPI automatically provides interactive API documentation through Swagger UI.

You can use it to:

* Explore available endpoints
* Test API requests
* Inspect request parameters
* View API responses

---

## 🔄 Application Workflow

```text
┌──────────────────────┐
│   Upload Documents   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Preprocess Documents │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│     Text Chunking    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Generate Embeddings  │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│    Check Cache       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Store / Reuse      │
│     Embeddings       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│     User Query       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Query Embedding      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Cosine Similarity    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Ranked Search Results│
└──────────────────────┘
```

---

## 🚀 Performance Optimization

The project is designed to reduce unnecessary computation through:

* ⚡ Embedding caching
* ✂️ Chunk-based processing
* 💾 Persistent local cache
* 🔎 Similarity-based retrieval
* ♻️ Reuse of previously generated embeddings

The caching layer is particularly useful when working with large documents or repeatedly searching the same document collection.

---

## 🔐 Git & Data Management

Sensitive and generated data should not be committed to GitHub.

The following directories/files should generally remain ignored:

```text
data/
cache/
*.db
*.pkl
*.index
.env
__pycache__/
venv/
```

This keeps the repository lightweight and prevents locally generated embeddings or private documents from being pushed accidentally.

---

## 🧪 Example Use Cases

This semantic search engine can be adapted for:

* 📄 Research paper search
* 📚 Knowledge-base search
* 🏢 Internal company documentation
* 📑 Legal document search
* 🏥 Healthcare document retrieval
* 💰 Financial document analysis
* 🎓 Educational content search
* 🤖 AI-powered document assistants

---

## 🔮 Future Improvements

Potential improvements include:

* 🔹 Vector database integration
* 🔹 RAG-based question answering
* 🔹 Metadata filtering
* 🔹 Hybrid keyword + semantic search
* 🔹 Document-level access control
* 🔹 Authentication and authorization
* 🔹 Cloud deployment
* 🔹 Background embedding generation
* 🔹 Search analytics
* 🔹 Improved document format support
* 🔹 Dockerized deployment

---

## 📸 Application Preview

Add screenshots of your Streamlit interface and API documentation here:

```text
docs/
├── streamlit-ui.png
├── api-docs.png
└── search-results.png
```

Example:

```markdown
![Streamlit Interface](docs/streamlit-ui.png)

![Search Results](docs/search-results.png)
```

---

## 📌 Key Technical Concepts

This project demonstrates practical implementation of:

* **Natural Language Processing**
* **Transformer Embeddings**
* **Semantic Search**
* **Vector Similarity**
* **Cosine Similarity**
* **Text Chunking**
* **Caching Strategies**
* **REST API Development**
* **FastAPI**
* **Streamlit**
* **Performance Optimization**

---

## 👨‍💻 Author

**Syed Sadain**

Python Full Stack Developer | Backend Developer | AI/ML Engineer

🔗 GitHub: https://github.com/syed-sadain

🔗 LinkedIn: https://www.linkedin.com/in/syed-sadain-a56ba827/

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---
