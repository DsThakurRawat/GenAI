import logging
import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
### CHROMADB PRO GUIDE ###

1. HIERARCHY (As seen in your notes):
   - Tenant: The top-level container (multi-user support).
   - Database: Isolated environments within a tenant.
   - Collection: Equivalent to a 'Table'. This is where we query.

2. PERSISTENCE:
   - Unlike simple stores, Chroma saves data to a local directory (sqlite/parquet).
   - If the directory exists, it loads the existing data instead of recreating.

3. CONCURRENCY:
   - Chroma handles multiple read/write requests using SQLite's locking mechanisms 
     when used in local mode, or via its server-client architecture in production.
"""

# Path where the vector store will be saved
PERSIST_DIRECTORY = "./chroma_db_storage"

def chromadb_full_lifecycle():
    try:
        # Cleanup previous run for demonstration
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            logger.info("Cleared old database directory.")

        # 1. Initialize Embeddings
        embeddings = OpenAIEmbeddings()

        # 2. Prepare Sample Data
        docs = [
            Document(page_content="Chroma is an open-source vector database.", metadata={"source": "manual"}),
            Document(page_content="Pinecone is a cloud-native vector database.", metadata={"source": "manual"}),
            Document(page_content="FAISS is a library for efficient similarity search.", metadata={"source": "manual"})
        ]

        # 3. Create Vector Store with Persistence
        logger.info("Creating Chroma collection and persisting to disk...")
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name="tech_comparison"
        )
        logger.info("Database successfully persisted.")

        # 4. Semantic Search (Similarity Search)
        query = "Tell me about open source databases"
        logger.info(f"Querying: {query}")
        
        results = vector_store.similarity_search(query, k=1)
        
        print("\n--- Search Result ---")
        for res in results:
            print(f"Content: {res.page_content}")
            print(f"Metadata: {res.metadata}")

        # 5. Loading from Disk (Demonstrating Persistence)
        logger.info("Demonstrating loading an existing database...")
        existing_db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name="tech_comparison"
        )
        print(f"Loaded database contains {len(existing_db.get()['ids'])} documents.")

    except Exception as e:
        logger.error(f"ChromaDB operation failed: {str(e)}")

if __name__ == "__main__":
    chromadb_full_lifecycle()
