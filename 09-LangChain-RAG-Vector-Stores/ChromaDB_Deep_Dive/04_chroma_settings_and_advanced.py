import logging
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# =============================================================================
# ADVANCED: DISTANCE METRICS AND PERFORMANCE SETTINGS
# =============================================================================
# This script dives into how Chroma actually CALCULATES similarity 
# and how to tune its performance for large datasets.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
###############################################################################
### DEEP DRILL: DISTANCE METRICS (THE MATH) ###
###############################################################################

When you search, Chroma calculates a "distance score". You can choose HOW:

1. COSINE SIMILARITY (Default: 'cosine')
   - Measures the angle between vectors. 
   - Great for text because it ignores the length of the document.
   - Score range: 0 (identical) to 1 (different).

2. EUCLIDEAN DISTANCE ('l2')
   - The straight-line distance between two points.
   - Best when the MAGNITUDE of the vector matters.

3. INNER PRODUCT ('ip')
   - Standard dot product. Often used for high-performance ranking.

### PERFORMANCE: HNSW ###
Chroma uses an algorithm called HNSW (Hierarchical Navigable Small Worlds) 
to search billions of vectors in milliseconds.
###############################################################################
"""

def advanced_chroma_config():
    try:
        # 1. Custom Settings (Persistence & Telemetry)
        # You can turn off telemetry if you have privacy concerns.
        logger.info("Configuring custom Chroma settings...")
        settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory="./chroma_advanced_config"
        )

        # 2. Creating a collection with a specific DISTANCE METRIC
        # We do this via the raw client first, then wrap it in LangChain.
        client = chromadb.PersistentClient(path="./chroma_advanced_config", settings=settings)
        
        collection_name = "euclidean_collection"
        logger.info(f"Creating collection '{collection_name}' with L2 distance...")
        
        # 'hnsw:space' is where you define the distance metric.
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "l2"} # Using Euclidean distance instead of Cosine
        )

        # 3. Wrapping in LangChain
        # LangChain can connect to this specifically configured collection.
        logger.info("Connecting LangChain to the custom L2 collection...")
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings()
        )
        
        print(f"Vector store is now using: {vector_store._collection.metadata['hnsw:space']}")

    except Exception as e:
        logger.error(f"Advanced config failed: {str(e)}")

if __name__ == "__main__":
    advanced_chroma_config()
