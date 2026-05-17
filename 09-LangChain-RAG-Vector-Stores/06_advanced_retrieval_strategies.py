import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

# =============================================================================
# EXPERT LEVEL: ADVANCED RETRIEVAL STRATEGIES
# =============================================================================
# This script covers how to solve the most common problem in RAG: 
# Finding the right balance between 'small chunks for search' 
# and 'large context for the LLM'.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
###############################################################################
### DEEP DIVE: THE PARENT DOCUMENT RETRIEVER ###
###############################################################################

1. THE PROBLEM:
   - If chunks are too SMALL: Search is accurate, but the LLM doesn't have enough 
     surrounding context to answer properly.
   - If chunks are too LARGE: Search is less accurate (noise), and you waste tokens.

2. THE SOLUTION (Parent Document Retrieval):
   - You store the LARGE documents ('Parents') in a Document Store.
   - You split those into SMALL chunks ('Children') and store them in a Vector Store.
   - When you search, you find the Small Children, but LangChain automatically 
     fetches the Large Parent for the LLM.

### OTHER STRATEGIES IN THIS SCRIPT:
- Multi-Vector Retrieval: Summarize a document and store the summary's vector, 
  but return the full text.
###############################################################################
"""

def parent_document_retrieval_demo():
    try:
        # 1. Setup our tools
        embeddings = OpenAIEmbeddings()
        
        # This splits the large parent documents
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        # This splits the parents into small children for better search
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        
        # 2. Setup storage
        # The Vector Store (Chroma) stores the small chunks
        vector_store = Chroma(
            collection_name="split_parents", 
            embedding_function=embeddings
        )
        # The Docstore (InMemory) stores the full parent documents
        store = InMemoryStore()
        
        # 3. Initialize the Retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        # 4. Add complex documents
        logger.info("Adding a large document as a 'Parent'...")
        large_doc = [
            Document(
                page_content="""
                CHAPTER 1: THE BEGINNING. 
                Long ago in a galaxy far away, there was a tiny planet. 
                (Imagine 2000 words of world-building here...)
                In the middle of this chapter, we mention the Secret Key: 'EXCALIBUR-99'.
                (Imagine another 1000 words of story here...)
                """,
                metadata={"title": "Planet X Guide"}
            )
        ]
        
        retriever.add_documents(large_doc, ids=None)
        
        # 5. Search for a tiny detail
        query = "What is the secret key?"
        logger.info(f"Searching for tiny detail: '{query}'")
        
        # This will find the small chunk containing 'EXCALIBUR-99'
        # BUT it will return the ENTIRE chapter (the parent).
        retrieved_docs = retriever.invoke(query)
        
        print("\n" + "="*40)
        print("PARENT DOCUMENT RETRIEVAL RESULT")
        print("="*40)
        print(f"Retrieved {len(retrieved_docs)} parent document(s).")
        print(f"Sample Content: {retrieved_docs[0].page_content[:200]}...")
        print("="*40)

    except Exception as e:
        logger.error(f"Advanced retrieval failed: {str(e)}")

def multi_vector_concept_explanation():
    """
    Expert Tip: Multi-Vector Retrieval.
    Sometimes the text is hard to embed (e.g., a table or image).
    Strategy:
    1. Use an LLM to generate a text SUMMARY of the table/image.
    2. Embed the SUMMARY in the vector store.
    3. Link that summary to the ORIGINAL table/image in the docstore.
    Result: You search by text description, but the LLM sees the original table.
    """
    print("\n[EXPERT TIP: MULTI-VECTOR RETRIEVAL]")
    print("Use summaries for embedding, but return the raw data for context.")

if __name__ == "__main__":
    parent_document_retrieval_demo()
    multi_vector_concept_explanation()
