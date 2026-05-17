# 🧠 Visual Mapping: Prototyping vs. Production RAG Architecture
*An architectural review auditing basic notebook retrieval flows against enterprise resilience, query rewriting pipelines, and dual-stage reranking matrices.*

---

## 🏗️ 1. Structural Comparison: Prototyping vs. Production Architecture

Basic notebooks implement single-threaded retrieval paths. Production pipelines decouple query semantic mapping from final generation nodes.

```mermaid
flowchart TD
    classDef proto fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fff;
    classDef prod fill:#022c22,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef logic fill:#1e293b,stroke:#cbd5e1,stroke-width:1px,color:#fff;

    subgraph Prototyping Path (rag_using_langchain.ipynb)
        Q1["Raw Input Query"] ::: logic --> Naive["Naive Similarity Retrieval (k=4)"] ::: proto
        Naive --> Gen1["Generation Model Prompt Window"] ::: proto
    end
    
    subgraph Enterprise Production Upgrades
        Q2["Raw Input Query"] ::: logic --> MultiQuery["Query Rewriter Node (Expands to 3-5 variants)"] ::: prod
        MultiQuery --> DenseSearch["Extract wide candidate pool (fetch_k=25)"] ::: logic
        DenseSearch --> MMRFilter["Maximal Marginal Relevance (MMR) Filtering"] ::: prod
        MMRFilter --> CrossEncoder["Cross-Encoder Reranking Scoring Node"] ::: prod
        CrossEncoder --> Gen2["Absolute Top-k Aligned Context Prompt Window"] ::: prod
    end
```

---

## 🔬 2. Paradigm Capability Scaling Matrix

| Functional Element | Prototyping Notebook Setup | Production Enterprise Standard |
| :--- | :--- | :--- |
| **Ingestion Resilience** | Fails silently if API connection drops or caps out. | **Graceful Failovers:** Caches clean raw text blocks immediately locally. |
| **Search Functionality** | Greedy Similarity (Fetches highly overlapping chunks). | **MMR Optimization:** Enforces strict candidate pool document diversity. |
| **Query Robustness** | Single point of failure on poorly worded user prompts. | **Multi-Query Routing:** Expands semantic vectors via LCEL chains. |
| **Context Chunking** | Pure Character limits (Breaks sentences randomly). | **Contextual Chunking:** Pre-injects document summaries before embedding. |

---

## 🔄 3. Dual-Stage Retrieval & Reranking Topology

Bi-encoders generate highly compressed dense representations optimized for speed. However, final candidate selection requires deep cross-attention keyword evaluation.

```mermaid
flowchart LR
    classDef stage fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef filter fill:#312e81,stroke:#a5b4fc,stroke-width:2px,color:#fff;

    Query["Expanded Search Targets"] --> BiEncoder["Stage 1: Dense Vector Index Search"] ::: stage
    BiEncoder --> Pool["Top 20 Broad Candidate Pool"] ::: filter
    
    Pool --> Reranker["Stage 2: Cross-Encoder Interleaved Attention Node"] ::: stage
    Query --> Reranker
    
    Reranker --> Final["Top 4 Precision-Scored Chunks Injected to Prompt Context"] ::: filter
```

---
*End of Module 10 Visual Mapping Reference Document.*
