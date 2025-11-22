import os
import hashlib
import pickle
import sqlite3
import datetime
from typing import List
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ============================
# CONFIG
# ============================
DATA_FOLDER = "data"
INDEX_FOLDER = "index"
INDEX_PATH = f"{INDEX_FOLDER}/documents.index"
META_PATH = f"{INDEX_FOLDER}/index_meta.pkl"
CACHE_DB = f"{INDEX_FOLDER}/embeddings_cache.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================
# UTIL FUNCTIONS
# ============================

def sha256_text(text: str) -> str:
    """Generate sha256 hash for caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def simple_tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"    \b[a-zA-Z]{2,}\b", text.lower())


# ============================
# EMBEDDER
# ============================

class Embedder:
    def __init__(self):
       self.model = SentenceTransformer(MODEL_NAME)


    def preprocess(self, text: str) -> str:
        import re
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def embed(self, text: str) -> np.ndarray:
        text = self.preprocess(text)
        emb = self.model.encode([text], convert_to_numpy=True)[0].astype("float32")
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


# ============================
# CACHE MANAGER
# ============================

class CacheManager:
    def __init__(self):
        os.makedirs(INDEX_FOLDER, exist_ok=True)
        self.conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                filename TEXT,
                hash TEXT,
                embedding BLOB,
                updated_at TEXT
            )
        """)
        self.conn.commit()

    def get(self, doc_id: str):
        row = self.conn.execute(
            "SELECT filename, hash, embedding FROM embeddings WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return None
        filename, hashv, emb_blob = row
        emb = pickle.loads(emb_blob)
        return filename, hashv, emb

    def set(self, doc_id: str, filename: str, hashv: str, emb: np.ndarray):
        emb_blob = pickle.dumps(emb)
        updated_at = datetime.datetime.utcnow().isoformat()

        self.conn.execute("""
            INSERT INTO embeddings(doc_id, filename, hash, embedding, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                filename=excluded.filename,
                hash=excluded.hash,
                embedding=excluded.embedding,
                updated_at=excluded.updated_at
        """, (doc_id, filename, hashv, emb_blob, updated_at))

        self.conn.commit()


# ============================
# BUILD INDEX
# ============================

def build_index():
    os.makedirs(INDEX_FOLDER, exist_ok=True)
    embedder = Embedder()
    cache = CacheManager()

    embeddings = []
    ids = []
    metadata = {}

    print("🔍 Building FAISS index...")

    for filename in sorted(os.listdir(DATA_FOLDER)):
        if not filename.endswith(".txt"):
            continue

        path = f"{DATA_FOLDER}/{filename}"
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
        clean_text = embedder.preprocess(text)
        hashv = sha256_text(clean_text)

        doc_id = os.path.splitext(filename)[0]

        cached = cache.get(doc_id)
        if cached and cached[1] == hashv:
            emb = cached[2]
            print(f"✓ Reused embedding for {doc_id}")
        else:
            emb = embedder.embed(text)
            cache.set(doc_id, path, hashv, emb)
            print(f"+ Computed embedding for {doc_id}")

        embeddings.append(emb)
        ids.append(doc_id)
        metadata[doc_id] = {
            "filename": filename,
            "preview": text[:200].replace("\n", " "),
            "length": len(text)
        }

    if not embeddings:
        raise ValueError("No documents in data/ folder.")

    X = np.vstack(embeddings)
    dim = X.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, INDEX_PATH)
    pickle.dump({"ids": ids, "meta": metadata}, open(META_PATH, "wb"))

    print("🎉 Index build complete!")


# ============================
# SEARCH
# ============================

def search_docs(query: str, top_k: int = 5):
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("Index not built yet. Call /reindex.")

    embedder = Embedder()
    q_emb = embedder.embed(query).reshape(1, -1)

    index = faiss.read_index(INDEX_PATH)
    meta_info = pickle.load(open(META_PATH, "rb"))

    ids = meta_info["ids"]
    meta = meta_info["meta"]

    scores, idxs = index.search(q_emb, top_k)

    results = []

    q_tokens = set(simple_tokenize(query))

    for score, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue

        doc_id = ids[i]
        info = meta[doc_id]

        doc_text = open(f"{DATA_FOLDER}/{info['filename']}", "r", encoding="utf-8").read()
        doc_tokens = set(simple_tokenize(doc_text))
        overlap = list(q_tokens & doc_tokens)

        results.append({
            "doc_id": doc_id,
            "score": float(score),
            "preview": info["preview"],
            "overlap_keywords": overlap[:10],
            "doc_length": info["length"]
        })

    return results


# ============================
# FASTAPI API
# ============================

app = FastAPI(title="Multi-document Embedding Search Engine with Caching")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/reindex")
def api_reindex():
    try:
        build_index()
        return {"status": "Index rebuilt successfully"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/search")
def api_search(req: SearchRequest):
    try:
        results = search_docs(req.query, req.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting API...")
    uvicorn.run("appx:app", host="0.0.0.0", port=8000, reload=True)
