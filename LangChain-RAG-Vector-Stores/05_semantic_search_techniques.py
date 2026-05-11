import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
### SEMANTIC SEARCH TECHNIQUES ###

1. SIMILARITY SEARCH:
   Finds the top 'k' documents with the smallest vector distance to the query.

2. SIMILARITY SEARCH WITH SCORE:
   Returns (Document, Score).
   - For Euclidean distance: Lower score is better.
   - For Cosine similarity: Higher score is better.

3. MMR (MAX MARGINAL RELEVANCE):
   Returns a diverse set of documents.
   It picks the top documents based on similarity, but then re-ranks them 
   to reduce redundancy.
"""

def search_techniques_demo():
    try:
        embeddings = OpenAIEmbeddings()
        
        # Data with some redundant information
        docs = [
            Document(page_content="The economy is growing by 5%."),
            Document(page_content="Economic growth has reached 5%."), # Redundant
            Document(page_content="Interest rates are expected to rise."),
            Document(page_content="Central banks may increase rates soon.") # Redundant
        ]
        
        # Create temporary in-memory store
        vector_store = Chroma.from_documents(docs, embeddings)

        query = "What is the state of the economy?"

        # 1. Similarity Search
        logger.info("Executing standard Similarity Search...")
        sim_results = vector_store.similarity_search(query, k=2)
        print("\n--- Similarity Search Results ---")
        for doc in sim_results:
            print(f"Content: {doc.page_content}")

        # 2. Similarity with Score
        logger.info("Executing Similarity Search with Score...")
        score_results = vector_store.similarity_search_with_score(query, k=2)
        print("\n--- Similarity Search with Score ---")
        for doc, score in score_results:
            print(f"Score: {score:.4f} | Content: {doc.page_content}")

        # 3. MMR (Diversity Search)
        logger.info("Executing MMR Search (for diversity)...")
        mmr_results = vector_store.max_marginal_relevance_search(query, k=2, fetch_k=4)
        print("\n--- MMR Search Results ---")
        for doc in mmr_results:
            print(f"Content: {doc.page_content}")

    except Exception as e:
        logger.error(f"Search demo failed: {str(e)}")

if __name__ == "__main__":
    search_techniques_demo()
