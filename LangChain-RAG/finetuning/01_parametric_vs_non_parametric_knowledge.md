# 🧠 Visual Mapping: Parametric vs. Non-Parametric Knowledge
*An architectural overview contrasting static internalized weights with dynamic retrieval-augmented memory access.*

---

## 🏛️ 1. Knowledge Internalization Mechanics

Foundation models store facts and reasoning capabilities through two distinct physical mechanisms.

```mermaid
flowchart TD
    classDef param fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#fff;
    classDef nonparam fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;

    Root["Foundation Model Knowledge Engine"]
    
    Root --> P["1. Parametric Knowledge"] ::: param
    P --> P1["Internalized inside parameter weights"] ::: param
    P --> P2["Updated via Next-Token Pre-Training & SFT"] ::: param
    P --> P3["Frozen static state after training cutoff"] ::: param
    
    Root --> NP["2. Non-Parametric Knowledge"] ::: nonparam
    NP --> NP1["External document & database stores"] ::: nonparam
    NP --> NP2["Injected dynamically at inference via RAG"] ::: nonparam
    NP --> NP3["Real-time, fully auditable citations"] ::: nonparam
```

---

## 🔬 2. Comparative Matrix: Parametric vs. Non-Parametric Access

| Feature | Parametric Knowledge | Non-Parametric Knowledge |
| :--- | :--- | :--- |
| **Physical Location** | Encoded inside matrix weight tensors. | External indices (Vector DBs, APIs, SQL Stores). |
| **Data Freshness** | Stale immediately post training cutoff. | Instantaneous real-time updates. |
| **Hallucination Rate**| High when forced to recall exact factual string sequences.| Extremely low (Bounded by strict context guardrails). |
| **Cost to Update** | Exceptionally high (Requires GPU clusters & retraining).| Negligible (Standard file embeddings or database writes). |
| **Auditability** | Zero verifiable citations or parent document tracking.| Native exact chunk and document reference logs. |

---

## 🔄 3. Execution Topology: Direct Query vs. RAG Injection

```mermaid
flowchart LR
    classDef fail fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fff;
    classDef success fill:#022c22,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef base fill:#1e293b,stroke:#cbd5e1,stroke-width:1px,color:#fff;

    subgraph Direct ["Direct Parametric Execution (Failure Risk)"]
        Q1["Input: Recent Private Event"] ::: base --> LLM1["Base Frozen Weights"] ::: base
        LLM1 --> Out1["Hallucination / Static Cutoff Refusal"] ::: fail
    end

    subgraph Augmented ["Non-Parametric Injection (RAG Grounding)"]
        Q2["Input: Recent Private Event"] ::: base --> Retrieve["Query External Vector Store"] ::: success
        Retrieve --> InjectedContext["Dynamic Ground Truth Fragments"] ::: success
        Q2 --> Merge["Prompt Composer"] ::: base
        InjectedContext --> Merge
        Merge --> LLM2["LLM Autoregressive Generation"] ::: base
        LLM2 --> Out2["Grounded Accurate Response"] ::: success
    end
```

---
*End of Module 01 Visual Mapping Reference Document.*
