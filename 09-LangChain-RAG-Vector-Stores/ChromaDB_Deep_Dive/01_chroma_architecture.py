import logging
import chromadb
from chromadb.config import Settings

# =============================================================================
# PRODUCTION LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
###############################################################################
### CHROMADB ARCHITECTURE: TENANCY & HIERARCHY ###
###############################################################################

Chroma is built to be multi-tenant. This means a single Chroma instance can 
serve multiple isolated users or environments.

1. TENANT (The Root):
   - At the highest level, we have a 'Tenant'.
   - By default, Chroma uses 'default_tenant'.
   - In a production SaaS app, each customer would be their own tenant.

2. DATABASE (Isolated Environments):
   - A single tenant can have multiple 'Databases'.
   - By default, Chroma uses the 'default_database'.
   - This is useful for separating environments like 'development', 'staging', 
     and 'production' within the same tenant.

3. COLLECTION (The Document Container):
   - Inside a database, we have 'Collections'.
   - A collection is like a Table in a relational database or a Folder.
   - Each collection has its own:
     - Name
     - Embedding Function (e.g., OpenAI, HuggingFace)
     - Metadata
     - Documents and IDs

4. DOCUMENT (The Data Unit):
   - The final level is the 'Document', which consists of:
     - page_content (The text)
     - metadata (Extra attributes like 'author', 'date', 'source')
     - embedding (The vector representation)
     - id (A unique identifier)
###############################################################################
"""

def explore_chroma_hierarchy():
    """
    Demonstrates how to interact with Chroma's hierarchy using the underlying 
    Chroma client. Note that LangChain's Chroma wrapper abstracts some of this, 
    but understanding the raw client is essential for advanced management.
    """
    try:
        # 1. Initialize the Persistent Client
        # PersistentClient saves data to disk immediately.
        client = chromadb.PersistentClient(path="./chroma_hierarchy_data")
        logger.info("Chroma Persistent Client initialized.")

        # 2. List Tenants (Advanced)
        # Note: In the local version, you usually work with the default tenant.
        logger.info(f"Current Tenant: {client.tenant}")

        # 3. Create a New Collection in the hierarchy
        collection_name = "curriculum_collection"
        logger.info(f"Creating/Getting collection: {collection_name}")
        
        # get_or_create_collection prevents errors if the script runs twice.
        collection = client.get_or_create_collection(name=collection_name)
        
        # 4. Add data to the collection
        logger.info("Adding documents to the collection...")
        collection.add(
            documents=["Chroma is multi-tenant.", "Chroma supports databases."],
            metadatas=[{"type": "arch"}, {"type": "arch"}],
            ids=["id1", "id2"]
        )

        # 5. Verify the hierarchy
        count = collection.count()
        logger.info(f"Collection '{collection_name}' now contains {count} items.")

    except Exception as e:
        logger.error(f"Failed to explore Chroma hierarchy: {str(e)}")

if __name__ == "__main__":
    explore_chroma_hierarchy()
