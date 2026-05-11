import logging
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# =============================================================================
# BEGINNER'S GUIDE: INTEGRATING CHROMADB WITH LANGCHAIN
# =============================================================================
# This script explains how LangChain acts as a "wrapper" or "bridge" between 
# your documents and the Chroma database.

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
###############################################################################
### CONCEPT FOR BEGINNERS: WHY LANGCHAIN + CHROMA? ###
###############################################################################

Imagine you have 1000 PDF files. 
1. You use LangChain to LOAD and SPLIT those PDFs into chunks.
2. You use an EMBEDDING MODEL (like OpenAI) to turn those chunks into numbers.
3. You need a place to STORE those numbers so you can search them later. 
   THAT PLACE IS CHROMA DB.

USE CASES:
- Chat with your documents (RAG).
- Search through internal company wikis.
- Customer support bots that "read" your help articles.
###############################################################################
"""

def langchain_chroma_basic_demo():
    """
    Step-by-step implementation for beginners.
    """
    try:
        # STEP 1: Initialize the Embedding Model
        # This model is the 'translator' that turns text into vectors.
        logger.info("Initializing OpenAI Embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # STEP 2: Create some 'Documents'
        # In LangChain, a Document is a piece of text + optional metadata.
        logger.info("Preparing sample documents...")
        docs = [
            Document(
                page_content="The first rule of Fight Club is: you do not talk about Fight Club.",
                metadata={"source": "movie_quotes", "year": 1999}
            ),
            Document(
                page_content="I'm going to make him an offer he can't refuse.",
                metadata={"source": "movie_quotes", "year": 1972}
            ),
            Document(
                page_content="May the Force be with you.",
                metadata={"source": "movie_quotes", "year": 1977}
            )
        ]

        # STEP 3: Create the Vector Store (Chroma)
        # from_documents() does 3 things automatically:
        #   a) Converts text to embeddings using the model.
        #   b) Creates a collection in Chroma.
        #   c) Saves the embeddings and text into Chroma.
        persist_directory = "./chroma_langchain_basics"
        logger.info(f"Creating Chroma store at {persist_directory}...")
        
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name="movie_masterpieces"
        )
        logger.info("Chroma store created and persisted.")

        # STEP 4: Perform a Similarity Search
        # This is the "magic" of RAG. We ask a question, and Chroma finds 
        # the most relevant quotes based on MEANING, not just keywords.
        query = "Which quote mentions a rule?"
        logger.info(f"Searching for: '{query}'")
        
        # k=1 means "Give me the single best match"
        results = vector_store.similarity_search(query, k=1)

        print("\n" + "="*30)
        print("SEARCH RESULTS")
        print("="*30)
        for doc in results:
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
        print("="*30)

    except Exception as e:
        logger.error(f"Error in basic demo: {str(e)}")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in your .env file!
    langchain_chroma_basic_demo()
