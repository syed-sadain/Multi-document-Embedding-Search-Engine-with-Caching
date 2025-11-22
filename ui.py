import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/search"

st.set_page_config(page_title="ML Document Search", layout="centered")

st.title("🔍 Multi-document Embedding Search Engine with Caching")
st.write("Search through 100+ ML concept documents using semantic search (FAISS + SentenceTransformer).")

query = st.text_input("Enter your query (example: 'neural networks', 'backpropagation', 'RNNs')")

top_k = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query!")
    else:
        with st.spinner("Searching..."):
            payload = {"query": query, "top_k": top_k}
            response = requests.post(API_URL, json=payload)

            if response.status_code != 200:
                st.error("API error! Is your FastAPI backend running?")
            else:
                results = response.json()["results"]

                st.success(f"Found {len(results)} results:")
                
                for item in results:
                    st.subheader(f"📄 {item['doc_id']}")
                    st.write(f"**Score:** {item['score']:.4f}")
                    st.write(f"**Preview:** {item['preview']}...")
                    st.write("---")
