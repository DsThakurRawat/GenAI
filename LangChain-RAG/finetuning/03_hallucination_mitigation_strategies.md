# 🧠 Visual Mapping: Hallucination Mitigation & Guardrail Frameworks
*An architectural reference mapping multi-stage verification chains, inference sampling parameter restrictions, and LLM-as-a-Critic self-correction loops.*

---

## 🏗️ 1. Multi-Vector Mitigation Topology

Eliminating hallucination vectors requires implementing parallel boundaries across retrieval, prompting logic, and sampling probability algorithms.

```mermaid
flowchart TD
    classDef boundary fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef logic fill:#1e293b,stroke:#cbd5e1,stroke-width:1px,color:#fff;

    Root["Comprehensive Mitigation Architecture"] ::: logic
    
    Root --> RAG["1. Non-Parametric Grounding"] ::: boundary
    RAG --> RAG1["Injects verified facts to replace guesswork"] ::: logic
    
    Root --> Sampling["2. Greedy Sampling Parameters"] ::: boundary
    Sampling --> S1["Temperature = 0.0 (Deterministic path)"] ::: logic
    Sampling --> S2["Top-P Constraint filtering"] ::: logic
    
    Root --> Prompt["3. Strict Guardrail Schemas"] ::: boundary
    Prompt --> P1["Explicit Fallback boundary instructions"] ::: logic
    
    Root --> Chains["4. Automated Critic Verification"] ::: boundary
    Chains --> C1["Multi-stage LLM-as-a-Critic parsing loops"] ::: logic
```

---

## 🔬 2. Visual Architecture: LLM-as-a-Critic Verification Loop

```mermaid
flowchart LR
    classDef input fill:#022c22,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef draft fill:#312e81,stroke:#a5b4fc,stroke-width:2px,color:#fff;
    classDef eval fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fff;

    Ctx["Source Ground Truth Context"] ::: input --> Gen["Draft Generation Node"] ::: draft
    Query["User Input Query"] ::: input --> Gen
    
    Gen --> DraftAns["Unverified Draft Answer"] ::: draft
    
    DraftAns --> Critic["Evaluator/Critic Node"] ::: eval
    Ctx --> Critic
    
    Critic --> Condition{"Contains Unsupported Extrapolations?"}
    
    Condition -- "Yes" --> Flag["Final Output: HALLUCINATION INTERCEPTED"] ::: eval
    Condition -- "No" --> Pass["Final Output: Audited Verified Response"] ::: input
```

---

## 📊 3. Sampling Probability Impact Analysis

| Sampling Setting | Probability Distribution Mechanics | Production Hallucination Impact |
| :--- | :--- | :--- |
| **High Temperature (e.g., 1.0)** | Flattens token probability distributions, forcing highly diverse sampling. | **Critical:** Forces logical hallucinations, factual divergence, and creative guesswork. |
| **Low Temperature (e.g., 0.0)** | Enforces greedy decoding, exclusively picking the peak probability token. | **Highly Resilient:** Guarantees absolute deterministic stability and contextual tracing. |

---
*End of Module 03 Visual Mapping Reference Document.*
