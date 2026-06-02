
# #📘  Multi-Document Embedding Search Engine with Caching

A semantic search engine that uses Machine Learning, NLP embeddings, and similarity search algorithms to retrieve the most relevant information across multiple documents. Includes a caching system to avoid repeated embedding generation and improve performance.

🚀 Features

Multi-document ingestion and preprocessing

Transformer-based embedding generation

Semantic search using cosine similarity

Efficient caching layer (avoids recomputation)

Fast and accurate AI-powered search results

Supports large documents through text chunking

Backend API + Streamlit UI

🧠 How It Works

Load multiple documents

Split into text chunks

Generate embeddings using ML models

Store embeddings in cache

User enters a query

Query embedding is compared with stored embeddings

Returns top relevant results based on similarity

🛠️ Tech Stack

Python

Embedding Models (Sentence Transformers / OpenAI)

NLP Preprocessing

Cosine Similarity

Pickle / SQLite DB Cache

Streamlit

FastAPI

📂 Project Structure
project/
│── src/
│── appx.py               # Backend server
│── ui.py                 # User Interface (Streamlit)
│── data/                 # Ignored by Git
│── cache/
│     ├── index_meta.pkl
│     ├── embeddings_cache.db
│     └── documents.index
│── README.md
│── requirements.txt
│── .gitignore

🖥️ How to Run the Project
✅ 1. Start Backend (Windows)
cd C:\Users\ssada\project
python appx.py


Backend must remain open and running.

✅ 2. Start User Interface
cd C:\Users\ssada\project
streamlit run ui.py

✅ 3. Start API (FastAPI)

API documentation available at:

👉 http://127.0.0.1:8000/docs

📦 Cache Files Stored Here

The system stores embeddings and metadata in:

index_meta.pkl

embeddings_cache.db

documents.index

These files allow fast loading without recomputing embeddings.

📦 Installation
pip install -r requirements.txt

▶️ Run the App

Streamlit UI:

streamlit run ui.py


Backend:

python appx.py






