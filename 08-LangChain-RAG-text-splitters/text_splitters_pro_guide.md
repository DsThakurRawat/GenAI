# ✂️ LangChain Text Splitters & Semantic Chunking Blueprint
*A robust reference guide mapping sliding structural overlap boundaries (`chunk_size`, `chunk_overlap`), hierarchical string array splits, and advanced cosine distance semantic clustering matrices.*

---

## 📐 1. Geometric Boundaries: Chunk Size vs. Overlap

Text Splitters maintain contextual thread continuity across truncated boundaries by employing sliding structural windows. 

```mermaid
graph LR
    classDef default fill:#1e293b,stroke:#cbd5e1,stroke-width:1px,color:#fff;
    classDef overlap fill:#312e81,stroke:#a5b4fc,stroke-width:2px,color:#fff;

    subgraph Unbroken Original Document Text
        A["[Token 0 ... Token 800]"] --> B["[Token 801 ... Token 1000]"] ::: overlap --> C["[Token 1001 ... Token 1800]"]
    end
    
    subgraph Chunk Slice 1: chunk_size=1000
        A --> B
    end
    
    subgraph Chunk Slice 2: chunk_overlap=200 sliding window
        B --> C
    end
```

### 🧠 Processing Tradeoffs:
- **Small Chunk Sizes (e.g., 256 tokens)**: Yields highly specific, clean cosine vector target queries. Minimizes context window consumption but risks starving downstream attention networks of required surrounding premise data.
- **Large Chunk Sizes (e.g., 2048 tokens)**: Delivers dense background narratives. Optimal for complex cognitive rationalization but risks injecting semantic noise and triggering cloud token limits.
- **Sliding Overlap**: Prevents destructive breaks mid-sentence. Set systematically at **10% to 20%** of target size parameters.

---

## 🏛️ 2. Structural Parsing vs. Pure Characters

```mermaid
graph TD
    classDef text fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef code fill:#022c22,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef semantic fill:#312e81,stroke:#a5b4fc,stroke-width:2px,color:#fff;

    Root["TextSplitter Implementation Protocol"]
    
    Root --> Rec["1. RecursiveCharacterTextSplitter"] ::: text
    Rec --> RecLogic["Iterates separator lists: ['\\n\\n', '\\n', ' ', '']<br/>Preserves sentence integrity."] ::: text
    
    Root --> Code["2. Language-Specific Code Splitter"] ::: code
    Code --> CodeLogic["Parses source AST syntax separators.<br/>Keeps classes and functions intact."] ::: code
    
    Root --> Sem["3. SemanticChunker"] ::: semantic
    Sem --> SemLogic["Evaluates dense sentence-level embedding vectors.<br/>Splits strictly when Cosine Distance triggers variance threshold."] ::: semantic
```

---

## 📊 3. Comprehensive Execution Frameworks Matrix

| Splitter Engine Class | Native Delimiter Strategy | Structural Retention Payload | Optimal Production Use Case Target |
| :--- | :--- | :--- | :--- |
| **`RecursiveCharacterTextSplitter`** | Hierarchical static list sweeps. | Plain text string content. | The universal industry default for standard natural language documents. |
| **`TokenTextSplitter`** | Exact `tiktoken` encoder counts. | Strict token bounded slices. | Budget optimization passing context directly to pricing models. |
| **`MarkdownHeaderTextSplitter`** | Markdown markers (`#`, `##`). | Injects section titles into chunk metadata keys. | Ingesting structured technical engineering documentation repositories. |
| **`SemanticChunker`** | Cosine similarity dips. | Thematically uniform groupings. | High-precision scientific knowledge bases where section sizes vary wildly. |

---

## 🔍 4. Advanced Semantic Chunker Deep Dive

Standard character chunkers split text blindly based on character length limits. The **`SemanticChunker`** uses an entirely distinct spatial clustering approach:

```mermaid
graph LR
    classDef default fill:#1e293b,stroke:#cbd5e1,stroke-width:1px,color:#fff;
    classDef break fill:#7f1d1d,stroke:#fca5a5,stroke-width:2px,color:#fff;

    S1["Sentence 1"] --> S2["Sentence 2"] --> S3["Sentence 3"] ::: break --> S4["Sentence 4"]
    
    Note over S1,S2: Low Cosine Distance<br/>(Identical Subject)
    Note over S2,S3: High Cosine Distance Dip<br/>(Subject shift triggers split)
```

### ⚙️ Mechanics:
1. Every individual sentence is embedded into dense spatial vectors dynamically.
2. The distance metrics between neighboring sentences are calculated continuously.
3. Chunks are spliced strictly at boundary points where semantic divergence exceeds percentile threshold settings.

---

## 📁 5. Reference Demonstration Scripts Syllabus
Run files inside this folder directly to test string parsing parameters:
- `01_understanding_chunking.py`: Visual overlap matrix inspections.
- `02_recursive_character_splitter.py`: Implementing default natural language pipelines.
- `03_token_text_splitter.py`: Bounding output arrays to raw byte encoding limits.
- `04_markdown_header_splitter.py`: Preserving document header hierarchies.
- `05_semantic_chunking.py`: Executing cosine spatial boundary split logic.
- `06_code_splitter.py`: Splitting programming syntax modules cleanly.
- `07_html_splitter.py`: Parsing HTML tags into structural child chunks.
- Niche scripts: `length_based.py`, `markdown_splitting.py`, `python_code_splitting.py`.
