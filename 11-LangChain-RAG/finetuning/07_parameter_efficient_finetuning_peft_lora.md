# 🧠 Visual Mapping: PEFT & Low-Rank Adaptation (LoRA) Architecture
*An architectural overview mapping rank decomposition matrices, parallel forward execution graphs, and quantized 4-bit memory optimizers.*

---

## 🏗️ 1. Parallel Injection Topology

LoRA bypasses the catastrophic cost of mutating massive pre-trained weight matrices by running secondary low-rank pathways directly alongside dense layers.

```mermaid
flowchart LR
    classDef frozen fill:#1f2937,stroke:#6366f1,stroke-width:2px,color:#fff;
    classDef train fill:#022c22,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef merge fill:#312e81,stroke:#a5b4fc,stroke-width:2px,color:#fff;

    Input["Layer Input Array (x)"] --> Static["Frozen Base Weights: W (d x d)"] ::: frozen
    
    subgraph LoRA Adapter Pathway
        Input --> MatA["Trainable Matrix A: d -> r"] ::: train
        MatA --> MatB["Trainable Matrix B: r -> d"] ::: train
    end
    
    Static --> Sum["Vector Addition Engine (+)"] ::: merge
    MatB --> Sum
    
    Sum --> Out["Activated Output Vector"]
```

---

## 🔬 2. Quantitative Footprint Scaling: Full SFT vs. QLoRA

| Hardware Metrics | Full Fine-Tuning (16-bit) | Quantized LoRA (4-bit Base + 16-bit LoRA) |
| :--- | :--- | :--- |
| **VRAM for 7B Model** | ~120 GB (Requires cluster of A100 GPUs) | **~6 GB** (Fully deployable on consumer RTX GPUs) |
| **Gradients Stored**| 100% of internal model parameter layers | **<2%** (Exclusively rank projection tensors) |
| **Optimizer States**| Massive (AdamW tracks mom/var across all weights)| Extremely compressed |
| **Risk Profile** | High catastrophic forgetting exposure | Preserves baseline model logic safely |

---

## 🔄 3. Inference-Time Weight Merging Mechanics

During offline training, adapter paths calculate gradients parallel to frozen layers. Prior to production server deployment, weights are mathematically collapsed into standard base checkpoints to achieve maximum throughput speed.

```mermaid
flowchart TD
    classDef offline fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef prod fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#fff;

    subgraph Offline Fine-Tuning Execution
        W["Frozen Weight Checkpoint (W)"] ::: offline
        L["Trained Adapter Weights (B @ A)"] ::: offline
    end
    
    W --> Sum["Matrix Add: W_merged = W + (B @ A)"]
    L --> Sum
    
    Sum --> Prod["Unified Single Checkpoint Deployment"] ::: prod
    Prod --> Edge["High-Speed Autoregressive Inference (Zero Latency Penalty)"] ::: prod
```

---
*End of Module 07 Visual Mapping Reference Document.*
