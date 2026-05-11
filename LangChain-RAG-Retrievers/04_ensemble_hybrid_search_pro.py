import logging
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
###############################################################################
### ADVANCED: ENSEMBLE RETRIEVER (HYBRID SEARCH) ###
###############################################################################

1. THE LOGIC:
   - Vector Search (FAISS/Chroma): Great at finding MEANING (Semantic).
   - BM25 Search: Great at finding EXACT KEYWORDS (Keyword).
   - Ensemble Retriever: Combines both into a single ranked result.

2. WHY USE IT?
   - Sometimes users search for specific jargon or IDs (e.g., "Error code 404"). 
     Vector search might find "Generic error handling", but BM25 will 
     find the exact "404" document.

3. WEIGHTS:
   - You can assign weights (e.g., 0.5 to Vector, 0.5 to BM25).
###############################################################################
"""

def ensemble_retrieval_demo():
    try:
        embeddings = OpenAIEmbeddings()
        
        docs = [
            Document(page_content="Deep learning is a subset of machine learning."),
            Document(page_content="The iPhone 15 was released in 2023."),
            Document(page_content="Apple computers use macOS.")
        ]

        # 1. Setup Keyword Retriever (BM25)
        logger.info("Initializing BM25 Retriever (Keyword)...")
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 2

        # 2. Setup Vector Retriever (Chroma)
        logger.info("Initializing Chroma Retriever (Semantic)...")
        vectorstore = Chroma.from_documents(docs, embeddings)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # 3. Create Ensemble Retriever
        # We combine them 50/50
        logger.info("Creating EnsembleRetriever (Hybrid Search)...")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

        # 4. Search
        # This will work well for both 'Apple' (keyword) and 'Subset' (semantic)
        query = "iPhone"
        logger.info(f"Hybrid Search Query: {query}")
        
        results = ensemble_retriever.invoke(query)

        print("\n" + "="*30)
        print("HYBRID SEARCH RESULTS")
        print("="*30)
        for doc in results:
            print(f"Content: {doc.page_content}")

    except Exception as e:
        logger.error(f"Ensemble search failed: {str(e)}")

if __name__ == "__main__":
    ensemble_retrieval_demo()
