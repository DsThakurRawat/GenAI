import logging
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# =============================================================================
# PRODUCTION LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
###############################################################################
### BEGINNER'S GUIDE: RETRIEVER BASICS ###
###############################################################################

1. WHAT IS A RETRIEVER?
   - A retriever is an interface that returns documents based on a query.
   - It is more flexible than a Vector Store because it can be ANY source 
     (Wikipedia, Arxiv, or even a SQL database).

2. SEARCH TYPES:
   - 'similarity' (Default): Finds the closest vectors.
   - 'mmr' (Max Marginal Relevance): Finds similar BUT diverse results.
   - 'similarity_score_threshold': Only returns results above a certain confidence.
###############################################################################
"""

def basic_retrieval_demo():
    try:
        # 1. Setup Data
        embeddings = OpenAIEmbeddings()
        docs = [
            Document(page_content="The weather in London is mostly rainy.", metadata={"loc": "UK"}),
            Document(page_content="Paris is known as the City of Light.", metadata={"loc": "FR"}),
            Document(page_content="London is the capital of the United Kingdom.", metadata={"loc": "UK"}),
        ]
        
        # 2. Create Vector Store
        vectorstore = Chroma.from_documents(docs, embeddings)

        # 3. DIFFERENT RETRIEVAL STRATEGIES
        
        # A. Basic Similarity (Top 2 results)
        logger.info("Executing Basic Similarity Retrieval...")
        basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        results_a = basic_retriever.invoke("Tell me about London")
        
        # B. MMR (Diversity Search)
        # Perfect for avoiding redundant results.
        logger.info("Executing MMR Retrieval (Diversity)...")
        mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 3})
        results_b = mmr_retriever.invoke("Tell me about London")

        print("\n" + "="*30)
        print("BASIC RETRIEVAL RESULTS")
        print("="*30)
        for doc in results_a:
            print(f"Content: {doc.page_content}")
            
    except Exception as e:
        logger.error(f"Basic retrieval failed: {str(e)}")

if __name__ == "__main__":
    basic_retrieval_demo()
