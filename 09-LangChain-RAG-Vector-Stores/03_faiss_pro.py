import logging
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
### FAISS PRO GUIDE ###

1. WHAT IT IS:
   Facebook AI Similarity Search (FAISS) is a library for fast search of dense vectors.
   It is NOT a database; it is an index.

2. WHY USE IT?
   - Speed: Extremely optimized for large datasets in local memory.
   - Prototyping: No setup needed (just pip install faiss-cpu).

3. LIMITATIONS:
   - No automatic persistence (you must manually save the index to a file).
   - No built-in authentication or multi-tenancy.
"""

def faiss_lifecycle_demo():
    try:
        embeddings = OpenAIEmbeddings()

        docs = [
            Document(page_content="RAG improves LLM accuracy by providing context."),
            Document(page_content="Embeddings convert text into dense vectors.")
        ]

        # 1. Create FAISS index
        logger.info("Creating FAISS index in-memory...")
        vector_store = FAISS.from_documents(docs, embeddings)

        # 2. Save index to local disk
        save_path = "faiss_index_store"
        logger.info(f"Saving FAISS index to {save_path}...")
        vector_store.save_local(save_path)

        # 3. Load index from local disk
        logger.info("Loading FAISS index from disk...")
        new_vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

        # 4. Search
        query = "How to improve accuracy?"
        results = new_vector_store.similarity_search(query)
        
        print(f"\nQuery: {query}")
        print(f"Top Result: {results[0].page_content}")

    except Exception as e:
        logger.error(f"FAISS operation failed: {str(e)}")

if __name__ == "__main__":
    faiss_lifecycle_demo()
