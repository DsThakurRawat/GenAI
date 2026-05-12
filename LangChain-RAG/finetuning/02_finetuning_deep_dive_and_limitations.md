# 🧠 Visual Mapping: Fine-Tuning Architectures & Production Limitations
*A comprehensive framework mapping model retraining schemas, Parameter-Efficient adaptations (LoRA), and training dataset topologies.*

---

## 🏗️ 1. Taxonomy of Model Customization Pipelines

```mermaid
flowchart TD
    classDef main fill:#1e293b,stroke:#cbd5e1,stroke-width:1px,color:#fff;
    classDef peft fill:#312e81,stroke:#a5b4fc,stroke-width:2px,color:#fff;
    classDef full fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fff;

    Root["Fine-Tuning Methodology"] ::: main
    
    Root --> F["Full-Parameter Fine-Tuning"] ::: full
    F --> F1["Updates all matrix layers"] ::: full
    F --> F2["High hardware memory requirements"] ::: full
    
    Root --> P["Parameter-Efficient Fine-Tuning (PEFT)"] ::: peft
    P --> LoRA["Low-Rank Adaptation (LoRA)"] ::: peft
    P --> QLoRA["Quantized LoRA (QLoRA 4-bit)"] ::: peft
    
    Root --> IT["Instruction Fine-Tuning"] ::: main
    IT --> IT1["Aligns completion models to dialogue flows"] ::: main
    
    Root --> DA["Domain Adaptation"] ::: main
    DA --> DA1["Continued unsupervised pre-training on domain text"] ::: main
```

---

## 🔬 2. Visual Architecture: Low-Rank Adaptation (LoRA) Mechanics

Standard fine-tuning computes updates for massive base parameter arrays. LoRA freezes these dense base layers and injects highly compressed rank decomposition matrices directly parallel to attention paths.

```mermaid
flowchart LR
    classDef base fill:#1f2937,stroke:#6366f1,stroke-width:2px,color:#fff;
    classDef lora fill:#022c22,stroke:#34d399,stroke-width:2px,color:#fff;

    Input["Layer Input Vector (d)"] --> Base["Frozen Base Model Weights (W)"] ::: base
    Input --> Down["Trainable Low-Rank Matrix A (d -> r)"] ::: lora
    Down --> Up["Trainable Low-Rank Matrix B (r -> d)"] ::: lora
    
    Base --> Add["Vector Summation (+)"]
    Up --> Add
    
    Add --> Output["Layer Output Activation Vector"]
```

> [!TIP]
> **Rank ($r$) Compression Impact:** Setting $r=8$ or $r=16$ enables parameter reduction footprints exceeding **98%**, allowing frontier model training directly on single edge consumer workstations without hitting CUDA Out-Of-Memory bounds.

---

## 📊 3. SFT Training Dataset Topology (JSONL Layout)

Instructional alignment pipelines require parsing unformatted natural prose into explicit multi-role chat JSON objects.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a professional corporate sentiment analysis classifier."
    },
    {
      "role": "user",
      "content": "The platform interface dashboard loaded instantly without errors."
    },
    {
      "role": "assistant",
      "content": "CLASSIFICATION: POSITIVE"
    }
  ]
}
```

---
*End of Module 02 Visual Mapping Reference Document.*
