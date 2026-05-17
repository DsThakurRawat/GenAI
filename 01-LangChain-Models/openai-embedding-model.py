from langchain_openai import OpenAIEmbeddings
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

3. EMBEDDING MODEL - e.g., 'text-embedding-3-small'
   - Input: String (Text)
   - Output: List of Floats (Vector)
   - Behavior: It DOES NOT generate text. It converts text into a mathematical 
     representation of its 'meaning'. Used for search, clustering, and RAG.
"""

# CONCEPTUAL: What are Embeddings?
"""
Embeddings are a way to represent text as a list of numbers (a vector).
1. Vector Space: Similar meanings are placed closer together in a high-dimensional space.
2. Semantic Search: Unlike keyword search (looking for exact words), embeddings allow 
   us to search by 'meaning'.
3. Math: We usually calculate the 'distance' between vectors using Cosine Similarity.
"""

"""
OPENAI EMBEDDING MODEL DEMONSTRATION
====================================

This script demonstrates how to use OpenAI's Embedding models to convert text into 
mathematical vectors for semantic search.

### PARAMETER IMPACT GUIDE (Docstring Explanation):

1. **model (str)**:
   - `text-embedding-3-small`: Current standard. Very cheap and effective (1536 dimensions).
   - `text-embedding-3-large`: More detailed, better for complex relationships (3072 dimensions).
   - `text-embedding-ada-002`: Legacy standard. Solid but superseded by the '3' series.

2. **dimensions (int)**:
   - *Note*: Only supported by 'text-embedding-3' models.
   - *Impact*: Allows you to "truncate" the vector to a smaller size (e.g., 512) without 
     losing much accuracy. Smaller vectors save database space and speed up search.
"""

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    api_key=settings.OPENAI_API_KEY,
    
    # PARAMETER: model
    # IMPACT: Determines the 'intelligence' and length of the resulting vector.
    model="text-embedding-3-small",
    
    # PARAMETER: dimensions (Optional for '3' series)
    # IMPACT: Allows reducing the vector size for efficiency.
    # dimensions=512 
)

def run_embedding_demo():
    try:
        # Example 1: Embedding a single piece of text
        text = "LangChain is a framework for building LLM applications."
        vector = embeddings.embed_query(text)
        
        logger.info(f"Generated embedding for: '{text}'")
        logger.info(f"Vector length: {len(vector)}") # Usually 1536 for OpenAI
        print(f"First 5 numbers of the vector: {vector[:5]}")

        # Example 2: Semantic Similarity (The 'Concept')
        # We compare a 'query' to multiple 'documents'
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
            # Formula: (A . B) / (||A|| * ||B||)
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            )
            
            print(f"\nSimilarity to: '{doc}'")
            print(f"Score: {similarity:.4f}")

        logger.success("Embedding similarity analysis complete.")

    except Exception as e:
        logger.error(f"Embedding Error: {e}")

if __name__ == "__main__":
    run_embedding_demo()