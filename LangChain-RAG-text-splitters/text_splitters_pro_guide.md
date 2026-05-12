# Mastering LangChain Text Splitters

Text Splitters are critical for building effective RAG (Retrieval-Augmented Generation) pipelines. They transform large, unwieldy documents into smaller, manageable "chunks" that fit within an LLM's context window.

## 1. Core Concepts: Size vs. Overlap

### 📏 Chunk Size
The maximum number of units (characters or tokens) in a single chunk.
- **Small Chunks (e.g., 200)**: Very specific, fast retrieval, but might lose surrounding context.
- **Large Chunks (e.g., 1000)**: Rich context, better for complex reasoning, but risk of "noise" and hitting token limits.

### 🔄 Chunk Overlap
The amount of text shared between adjacent chunks.
- **Purpose**: To maintain continuity. Without overlap, a critical sentence might be cut in half, destroying its meaning.
- **Best Practice**: Usually 10-20% of the `chunk_size` (e.g., Size=1000, Overlap=200).

---

## 2. Frequently Used Splitters

| Splitter Type | Best For | Logic |
| :--- | :--- | :--- |
| **Recursive Character** | Plain Text | Splits on `\n\n`, then `\n`, then ` `, then `""`. Preserves paragraphs/sentences. |
| **Token Splitter** | LLM Accuracy | Counts tokens (using `tiktoken`). Most accurate for context window management. |
| **Markdown Header** | Documentation | Splits on `#`, `##`, etc. Adds structure to metadata. |
| **Code Splitter** | Programming | Knows syntax rules for Python, JS, etc. Keeps functions intact. |
| **HTML Header** | Web Scraping | Understands `<h1>`, `<h2>` tags. |
| **Semantic Chunker** | Deep Meaning | Uses embeddings to group sentences by topic similarity. |

---

## 🚀 When to Use Which?

1. **Standard Articles/PDFs**: Always start with `RecursiveCharacterTextSplitter`. It's the most reliable for natural language.
2. **Technical Docs (.md)**: Use `MarkdownHeaderTextSplitter` first to get the structure, then use `Recursive` to break down large sections.
3. **Optimizing for OpenAI/Claude**: Use `TokenTextSplitter` to ensure you aren't wasting money on overly large chunks.
4. **Codebases**: Use `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` to keep logic blocks together.
5. **High-Precision RAG**: Use `SemanticChunker`. It's slower but ensures every chunk is about a single topic.

---

## 📁 Repository Guide
I have created detailed scripts with production-quality logging for each:
- `01_understanding_chunking.py`: Visualizing Size and Overlap.
- `02_recursive_character_splitter.py`: The industry standard.
- `03_token_text_splitter.py`: Accuracy for LLM tokens.
- `04_markdown_header_splitter.py`: Splitting by document structure.
- `05_semantic_chunking.py`: Advanced meaning-based grouping.
- `06_code_splitter.py`: Syntax-aware splitting for code.
- `07_html_splitter.py`: Structural splitting for web content.
