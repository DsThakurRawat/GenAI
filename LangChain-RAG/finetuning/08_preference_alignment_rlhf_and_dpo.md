# 🧠 Visual Mapping: Human Preference Alignment (RLHF vs. DPO)
*An architectural review mapping the multi-stage reinforcement learning from human feedback loops alongside direct preference cross-entropy minimization paths.*

---

## 🏗️ 1. Classical RLHF Execution Lifecycle (3-Stage Framework)

Reinforcement Learning from Human Feedback uses a secondary neural network to evaluate generations, mutating active baseline checkpoints via dynamic policy updates.

```mermaid
flowchart TD
    classDef stage fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef logic fill:#1e293b,stroke:#cbd5e1,stroke-width:1px,color:#fff;
    classDef ppo fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fff;

    subgraph Stage 1: SFT Baseline Alignment
        Raw["Raw Unaligned Foundation Model"] ::: logic --> SFT["Supervised Fine-Tuning (SFT) over dialogue sets"] ::: logic
        SFT --> Base["SFT Checkpoint Base Model"] ::: stage
    end
    
    subgraph Stage 2: Standalone Reward Modeling
        Prompts["Prompt Sets"] ::: logic --> GenA["Generate Completion Alpha"] ::: logic
        Prompts --> GenB["Generate Completion Beta"] ::: logic
        GenA --> Human["Human Preference Ranking Labeling"] ::: logic
        GenB --> Human
        Human --> TrainRM["Train Reward Model to output scalar values"] ::: stage
    end
    
    subgraph Stage 3: Proximal Policy Optimization (PPO)
        Base --> GenActive["Active LLM Generation Loop"] ::: logic
        GenActive --> EvalRM["Reward Model evaluates output trajectory"] ::: logic
        EvalRM --> Scalar["Scalar Value (e.g., +1.2 or -0.8)"] ::: logic
        Scalar --> UpdatePPO["PPO loss optimization algorithm"] ::: ppo
        UpdatePPO --> MutatePolicy["Mutate active LLM parameter matrices"] ::: ppo
        MutatePolicy --> SafeModel["Highly Secure Aligned Production Model"] ::: stage
    end
```

---

## ⚡ 2. Direct Preference Optimization (DPO) Topology

Direct Preference Optimization skips the complex reward evaluation stage entirely. It maps policy updates directly against verified target pairings.

```mermaid
flowchart LR
    classDef data fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef dpo fill:#312e81,stroke:#a5b4fc,stroke-width:2px,color:#fff;
    classDef target fill:#022c22,stroke:#34d399,stroke-width:2px,color:#fff;

    Dataset["Contrastive DPO Datasets"] ::: data --> Parse["Extract Pairings"] ::: data
    
    subgraph Contrastive Processing Buffer
        Parse --> C["Chosen Output Probability Target"] ::: target
        Parse --> R["Rejected Output Probability Target"] ::: data
    end
    
    C --> Loss["Compute Contrastive Cross-Entropy Loss"] ::: dpo
    R --> Loss
    
    Loss --> Backprop["Direct Backpropagation Over Base Parameters"] ::: dpo
    Backprop --> Final["Highly Stable Aligned Model Checkpoint"] ::: target
```

---

## 📊 3. Paradigm Efficiency Matrix

| Assessment Area | Classic RLHF (PPO) | Direct Preference Optimization (DPO) |
| :--- | :--- | :--- |
| **Architectural Models Required** | **3 Models:** SFT Base, Standalone Reward Model, Reference Model. | **1 Model:** Optimized directly against the static pre-trained base checkpoints. |
| **Compute Overhead** | Exceptionally high (PPO loads multiple models in VRAM). | **Highly efficient** (Requires ~50% of the training VRAM footprint). |
| **Optimization Stability** | Highly volatile; susceptible to reward hacking and looping. | Exceptionally stable; deterministic cross-entropy loss tracking. |
| **Output Grounding** | Forces polite stylistic outputs safely. | Preserves underlying factual logic mapping directly. |

---
*End of Module 08 Visual Mapping Reference Document.*
