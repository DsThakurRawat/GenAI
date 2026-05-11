# 🏹 LangChain RAG: Mastering Retrievers

This repository contains a comprehensive deep-dive into **Retrievers**, the specialized components that fetch relevant information for your LLM.

## 🧭 Roadmap

1. **`01_retriever_basics_pro.py`**: Understanding Similarity search vs. MMR (Diversity) search.
2. **`02_multi_query_retriever_pro.py`**: Using an LLM to generate multiple query variations to improve recall.
3. **`03_contextual_compression_pro.py`**: Post-processing documents to extract only relevant snippets and reduce noise.
4. **`04_ensemble_hybrid_search_pro.py`**: Combining Semantic search (Vector) and Keyword search (BM25).
5. **`05_parent_document_retriever_pro.py`**: Finding small chunks but returning large parent documents for context.
6. **`06_self_querying_pro.py`**: Teaching the LLM to write metadata filters based on natural language.
7. **`07_external_retrievers.py`**: Fetching live data from Wikipedia and Arxiv.

## 📊 Key Concepts

### Why do we need advanced retrievers?
Standard vector search (similarity) is powerful but limited. It can be fooled by word phrasing and often returns too much "noise" (irrelevant text).

- **Recall**: Finding ALL relevant documents. (Solved by Multi-Query).
- **Precision**: Returning ONLY the relevant parts. (Solved by Contextual Compression).
- **Hybrid Search**: Handling both meaning and exact keywords. (Solved by Ensemble).
- **Structured Search**: Handling dates, categories, and tags. (Solved by Self-Query).

---

## 🚀 Production Quality
All scripts in this folder include:
- `logging`: For tracking query expansion and retrieval steps.
- `Exception Handling`: To manage network errors and model timeouts.
- `Detailed Comments`: Explaining the "Why" and "How" for every step.
