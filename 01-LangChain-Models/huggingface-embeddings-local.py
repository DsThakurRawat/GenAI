from langchain_huggingface import HuggingFaceEmbeddings
from src.core.logger import logger
from src.core.config import settings
import numpy as np

# CONCEPTUAL: LLMs vs. Chat Models vs. Embedding Models
"""
1. LLM (Large Language Model) - e.g., 'text-davinci-003' (Legacy)
   - Input: String (Prompt)
   - Output: String (Completion)
   - Behavior: Pure text completion. If you say "The capital of India is", 
     it predicts "New Delhi". It doesn't understand "Chat" roles naturally.

2. CHAT MODEL - e.g., 'gpt-4o', 'claude-3.5-sonnet'
   - Input: List of Message Objects (System, Human, AI)
   - Output: Message Object (AIMessage)
   - Behavior: Optimized for conversation. It understands context, history, 
     and specific instructions (System Prompts). 
     *Note: Most modern LangChain projects use Chat Models.*

3. EMBEDDING MODEL - e.g., 'sentence-transformers/all-MiniLM-L6-v2'
   - Input: String (Text)
   - Output: List of Floats (Vector)
   - Behavior: It DOES NOT generate text. It converts text into a mathematical 
     representation of its 'meaning'. Used for search, clustering, and RAG.
"""

# CONCEPTUAL: Local Embeddings & Quantization
"""
Running embeddings locally is useful for privacy and cost.
- 'sentence-transformers' is the library for local embeddings.
- 'HuggingFaceEmbeddings' is LangChain's wrapper.
- We use small models (e.g., 'all-MiniLM-L6-v2') that fit on consumer hardware.
"""

"""
LOCAL HUGGINGFACE EMBEDDING DEMONSTRATION
=========================================

This script demonstrates how to run embedding models locally using 
`sentence-transformers` and LangChain.

### PARAMETER IMPACT GUIDE (Docstring Explanation):

1. **model_name (str)**:
   - `sentence-transformers/all-MiniLM-L6-v2`: Small, fast, and very popular for basic search.
   - `sentence-transformers/all-mpnet-base-v2`: Larger and more accurate, but slower.
   - `BAAI/bge-small-en-v1.5`: High performance on leaderboards for its size.

2. **model_kwargs (dict)**:
   - `device`: Set to `cuda` for GPU or `cpu`.
   - `trust_remote_code`: Necessary for some custom models on HF Hub.

3. **encode_kwargs (dict)**:
   - `normalize_embeddings`: If True, calculates the dot product becomes the same as 
     cosine similarity. Highly recommended for most search tasks.
"""

# Initialize HuggingFaceEmbeddings
# Note: This will download the model to your system (around 100-500MB)
embeddings = HuggingFaceEmbeddings(
    # PARAMETER: model_name
    # IMPACT: Determines the 'meaning' capture and vector size (e.g., 384 dimensions).
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    
    # PARAMETER: model_kwargs
    # IMPACT: Configures low-level model loading (e.g., trust_remote_code).
    model_kwargs={'trust_remote_code': True},
    
    # PARAMETER: encode_kwargs
    # IMPACT: Configures how the text is processed (e.g., normalization).
    encode_kwargs={'normalize_embeddings': True}
)

def run_local_embedding_demo():
    try:
        # Example 1: Embedding a single piece of text
        text = "LangChain is a framework for building LLM applications."
        vector = embeddings.embed_query(text)
        
        logger.info(f"Generated local embedding for: '{text}'")
        logger.info(f"Vector length: {len(vector)}") # Usually 384 or 768 for MiniLM
        print(f"First 5 numbers of the vector: {vector[:5]}")

        # Example 2: Semantic Similarity (Local)
        query = "How do I build AI apps?"
        
        docs = [
            "A guide to developing large language model applications.",
            "The weather in London is quite rainy today.",
            "Python programming for data science beginners."
        ]
        
        logger.info(f"Comparing query: '{query}' with {len(docs)} documents...")
        
        # Embed the query and the documents
        query_vector = np.array(embeddings.embed_query(query))
        
        for doc in docs:
            doc_vector = np.array(embeddings.embed_query(doc))
            
            # Calculate Cosine Similarity
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            )
            
            print(f"\nSimilarity to: '{doc}'")
            print(f"Score: {similarity:.4f}")

        logger.success("Local embedding similarity analysis complete.")

    except Exception as e:
        logger.error(f"Local Embedding Error: {e}")
        print("\n[TIP] Ensure 'sentence-transformers' is installed (`pip install sentence-transformers`).")

if __name__ == "__main__":
    run_local_embedding_demo()
