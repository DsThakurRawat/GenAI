from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from src.core.logger import logger
from src.core.config import settings
import numpy as np

# CONCEPTUAL: Vector Dimensions & Methods
"""
1. Dimensions: OpenAI's 'text-embedding-3' models allow you to specify dimensions.
   - Higher dimensions = More detail but slower/more expensive.
   - 300 dimensions is a great balance for efficiency.

2. embed_query vs. embed_documents:
   - .embed_query(): Used for a single string (the user's question).
   - .embed_documents(): Used for multiple strings (the knowledge base).
   *Note: Some providers use different models for query vs. documents.*
"""

"""
DOCUMENT SIMILARITY DEMONSTRATION
=================================

This script demonstrates how to compare a query against multiple documents using 
embeddings and cosine similarity.

### PARAMETER IMPACT GUIDE (Docstring Explanation):

1. **model**: 
   - `text-embedding-3-large`: High-performance embedding model.
   - **Impact**: Provides better semantic understanding than 'small' models.

2. **dimensions (int)**:
   - **Impact**: Reduces the size of the resulting vector. 
   - **Variation**: 300 is highly compact; 3072 is the maximum for this model.
   - **Decision**: Lower dimensions save storage and compute time with minimal accuracy loss.
"""

# Initialize Embedding Model
# We use 'text-embedding-3-large' as requested, with custom dimensions
embedding = OpenAIEmbeddings(
    api_key=settings.OPENAI_API_KEY,
    
    # PARAMETER: model
    # IMPACT: Determines the 'intelligence' of the meaning capture.
    model='text-embedding-3-large', 
    
    # PARAMETER: dimensions
    # IMPACT: Truncates the vector to 300 numbers instead of 3072.
    dimensions=300
)

def run_document_similarity_demo():
    try:
        documents = [
            "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
            "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
            "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
            "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
            "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
        ]

        query = 'Tell me about Jasprit Bumrah'
        
        logger.info(f"Query: {query}")
        logger.info(f"Computing embeddings for {len(documents)} documents...")

        # 1. Generate Embeddings
        doc_embeddings = embedding.embed_documents(documents)
        query_embedding = embedding.embed_query(query)

        # 2. Calculate Similarity Scores
        # sklearn's cosine_similarity returns a matrix; we take the first row [0]
        scores = cosine_similarity([query_embedding], doc_embeddings)[0]

        # 3. Find the best match
        best_index = np.argmax(scores)
        best_score = scores[best_index]

        logger.success("Similarity matching complete.")

        print("\n" + "="*30)
        print(f"SEARCH RESULTS")
        print("="*30)
        print(f"Query: {query}")
        print(f"Best Match: {documents[best_index]}")
        print(f"Similarity Score: {best_score:.4f}")
        print("="*30)

    except Exception as e:
        logger.error(f"Similarity Calculation Error: {e}")

if __name__ == "__main__":
    run_document_similarity_demo()