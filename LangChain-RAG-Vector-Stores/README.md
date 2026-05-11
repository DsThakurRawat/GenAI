# 📦 LangChain RAG: Vector Stores & Databases

This repository contains a deep-dive into Vector Stores and Databases for RAG applications.

## 🧭 Roadmap

1. **`01_embeddings_pro.py`**: Understanding how to map text to vector space using OpenAI and HuggingFace models.
2. **`02_chromadb_pro.py`**: Deep dive into ChromaDB persistence, collection hierarchy, and multi-tenancy.
3. **`03_faiss_pro.py`**: Local, high-performance similarity search with Facebook's FAISS library.
4. **`04_pinecone_pro.py`**: Production-grade cloud vector database management with Pinecone.
5. **`05_semantic_search_techniques.py`**: Advanced retrieval methods: Similarity Search, MMR (Max Marginal Relevance), and Distance Scoring.

## 🔑 Key Concepts

### Vector Store vs. Database
- **Vector Store**: A library (like FAISS) that indexes vectors. Good for local, static use.
- **Vector Database**: A full system (like Chroma or Pinecone) that adds **Persistence, Concurrency, and Security**.

### ChromaDB Hierarchy
Based on official architecture:
- **Tenant**: Top-level owner.
- **Database**: Isolation within a tenant.
- **Collection**: Where documents live (like a SQL table).

## 🚀 Semantic Search Logic
Semantic search works by calculating the "distance" between the query vector and the document vectors. 
- **Cosine Similarity**: Measures the angle between vectors (good for comparing text meaning regardless of length).
- **Euclidean Distance**: Measures the straight-line distance between points.
