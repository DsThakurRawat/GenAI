# 🧠 Visual Mapping: Pre-Training Paradigms & Continued Domain Adaptation
*An architectural reference mapping causal language modeling workflows, autoregressive loss computations, and unsupervised corpus ingestion sequences.*

---

## 🏗️ 1. Causal Next-Token Prediction Architecture

Self-Supervised pre-training processes unstructured inputs by continuously computing softmax probabilities over complete vocabulary indices.

```mermaid
flowchart TD
    classDef io fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef model fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#fff;
    classDef comp fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fff;

    Tokens["Input Sequence Context Window"] ::: io --> Embed["Dense Token Embedding Projections"] ::: model
    Embed --> Attn["Stacked Multi-Head Self-Attention Blocks"] ::: model
    Attn --> FF["Feed-Forward Parameter Layers"] ::: model
    FF --> Softmax["Softmax Probability Output Distribution"] ::: model
    
    Softmax --> Loss["Compute Cross-Entropy Loss against hidden Target Token"] ::: comp
    Loss --> Backprop["Backpropagate Gradients down dense base networks"] ::: comp
    Backprop --> Update["Mutate Base Model Parametric Weights"] ::: comp
```

---

## 🔬 2. Context Sliding Window Sequencing

To maximize GPU hardware processing throughput, raw sequences are unrolled dynamically across consecutive indexing loops.

```mermaid
flowchart LR
    classDef state fill:#1e293b,stroke:#cbd5e1,stroke-width:1px,color:#fff;
    classDef label fill:#022c22,stroke:#34d399,stroke-width:2px,color:#fff;

    subgraph Batch Step 1
        C1["Context: ['Foundation']"] ::: state --> L1["Target Label: 'models'"] ::: label
    end
    
    subgraph Batch Step 2
        C2["Context: ['Foundation', 'models']"] ::: state --> L2["Target Label: 'learn'"] ::: label
    end
    
    subgraph Batch Step 3
        C3["Context: ['Foundation', 'models', 'learn']"] ::: state --> L3["Target Label: 'representations'"] ::: label
    end
```

---

## 🔄 3. Domain Adaptation Pipeline Topology

Continued pre-training strips out human dialogue prompts entirely, formatting raw custom file archives into infinite sliding window token strings.

```mermaid
flowchart LR
    classDef raw fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef pipe fill:#312e81,stroke:#a5b4fc,stroke-width:2px,color:#fff;

    RawData["Raw Specialized Domain Files (Medical/Legal/Code)"] ::: raw --> Tokenize["BPE / Custom Tokenization Stripping"] ::: pipe
    Tokenize --> Packing["Max Sequence Packing (e.g., 4096 tokens)"] ::: pipe
    Packing --> UnsupTraining["Unsupervised Parameter Adaptation Loop"] ::: pipe
    UnsupTraining --> SpecWeights["Domain-Specialized Foundation Model"] ::: raw
```

---
*End of Module 06 Visual Mapping Reference Document.*
