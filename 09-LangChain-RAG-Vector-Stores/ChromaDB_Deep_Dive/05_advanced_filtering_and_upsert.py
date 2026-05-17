import logging
import chromadb
from chromadb.config import Settings

# =============================================================================
# EXPERT: ADVANCED FILTERING AND UPSERT STRATEGIES
# =============================================================================
# This script covers specific ChromaDB features found in the official documentation:
# 1. Upsert (Update if exists, else Insert)
# 2. Filtering by Document Content ($contains)
# 3. Complex logical filters ($and, $or)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
###############################################################################
### DEEP DRILL: UPSERT & TEXT FILTERING ###
###############################################################################

1. UPSERT:
   - In standard 'Add', if you add an ID that already exists, it might error or 
     create a duplicate.
   - 'Upsert' is smarter: it checks if the ID exists. If yes, it updates it. 
     If no, it creates it.

2. WHERE_DOCUMENT:
   - Standard 'where' filters by METADATA (e.g., date, author).
   - 'where_document' filters by the actual TEXT CONTENT of the document.
   - Example: Find documents that contain the word 'important'.

3. COMPLEX LOGIC:
   - You can combine multiple filters using nested dictionaries with $and/$or.
###############################################################################
"""

def advanced_chroma_features_demo():
    try:
        # Initialize client
        client = chromadb.PersistentClient(path="./chroma_expert_features")
        collection = client.get_or_create_collection(name="expert_collection")

        # 1. DEMONSTRATING UPSERT
        logger.info("Executing UPSERT operation...")
        # First time adding doc1
        collection.upsert(
            ids=["doc1"],
            documents=["This is the original text."],
            metadatas=[{"version": 1}]
        )
        # Second time - same ID, different text (this will update it)
        collection.upsert(
            ids=["doc1"],
            documents=["This is the UPDATED text via UPSERT."],
            metadatas=[{"version": 2}]
        )
        logger.info("Upsert successful. Version 2 is now stored.")

        # 2. ADDING DATA FOR FILTERING
        collection.add(
            ids=["doc2", "doc3"],
            documents=["The secret ingredient is honey.", "The secret ingredient is sugar."],
            metadatas=[{"category": "food", "healthy": True}, {"category": "food", "healthy": False}]
        )

        # 3. WHERE_DOCUMENT FILTER ($contains)
        logger.info("Searching with 'where_document' filter ($contains 'honey')...")
        results = collection.query(
            query_texts=["What is the secret?"],
            n_results=1,
            where_document={"$contains": "honey"} # Filters by text content!
        )
        print(f"\nText Filter Result: {results['documents'][0]}")

        # 4. COMPLEX LOGICAL FILTER ($and)
        logger.info("Searching with complex $and filter (category=food AND healthy=True)...")
        complex_results = collection.query(
            query_texts=["Give me healthy food secrets"],
            n_results=1,
            where={
                "$and": [
                    {"category": {"$eq": "food"}},
                    {"healthy": {"$eq": True}}
                ]
            }
        )
        print(f"Complex Filter Result: {complex_results['documents'][0]}")

    except Exception as e:
        logger.error(f"Expert feature demo failed: {str(e)}")

if __name__ == "__main__":
    advanced_chroma_features_demo()
