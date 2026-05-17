import logging
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# =============================================================================
# INTERMEDIATE: FILTERING AND MANAGING DATA (CRUD) IN CHROMA
# =============================================================================
# In a real app, your data changes. You need to add more, update old info, 
# or delete incorrect entries. You also need to filter searches.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
###############################################################################
### CONCEPT: METADATA FILTERING ###
###############################################################################

Chroma uses 'where' clauses to filter results. 
Example: "Find all quotes about 'Force' BUT ONLY from the year 1977."

Supported operators:
- $eq (Equal)
- $ne (Not Equal)
- $gt (Greater Than)
- $lt (Less Than)
- $and / $or (Logical combinations)
###############################################################################
"""

def chroma_crud_and_filtering():
    try:
        embeddings = OpenAIEmbeddings()
        persist_dir = "./chroma_advanced_management"
        
        # 1. INITIALIZE (Load existing or create new)
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="management_demo"
        )

        # 2. CREATE (ADD)
        # We assign explicit IDs so we can update/delete them later.
        logger.info("Adding documents with explicit IDs...")
        new_docs = [
            Document(page_content="Python is great for AI.", metadata={"lang": "python", "rating": 10}),
            Document(page_content="Javascript is great for Web.", metadata={"lang": "js", "rating": 8})
        ]
        ids = ["doc_py_01", "doc_js_01"]
        
        vector_store.add_documents(documents=new_docs, ids=ids)

        # 3. READ (FILTERED SEARCH)
        # Goal: Search only within 'python' language documents.
        query = "What is great for AI?"
        logger.info(f"Filtering search for '{query}' with lang=python...")
        
        filtered_results = vector_store.similarity_search(
            query, 
            k=1, 
            filter={"lang": "python"} # This is the 'where' clause
        )
        print(f"\nFiltered Result: {filtered_results[0].page_content}")

        # 4. UPDATE
        # If we want to change the content or metadata of an existing ID.
        logger.info("Updating existing document 'doc_py_01'...")
        updated_doc = Document(
            page_content="Python is the BEST for AI and Data Science.", 
            metadata={"lang": "python", "rating": 11}
        )
        vector_store.update_document(document_id="doc_py_01", document=updated_doc)

        # 5. DELETE
        # Removing a specific document by its ID.
        logger.info("Deleting 'doc_js_01'...")
        vector_store.delete(ids=["doc_js_01"])
        
        # Verify deletion
        logger.info(f"Collection now has {len(vector_store.get()['ids'])} documents.")

    except Exception as e:
        logger.error(f"Error in CRUD demo: {str(e)}")

if __name__ == "__main__":
    chroma_crud_and_filtering()
