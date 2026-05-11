import logging
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
### PINECONE PRO GUIDE ###

1. WHAT IT IS:
   A fully managed, cloud-native vector database. 
   Ideal for enterprise production (billions of vectors).

2. PRODUCTION FEATURES:
   - Authentication: Requires PINECONE_API_KEY.
   - Durability: Distributed architecture ensures data is never lost.
   - High Concurrency: Built to handle thousands of simultaneous queries.
   - Metadata Filtering: Efficiently filter by categories, tags, or IDs.
"""

def pinecone_demo():
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = "langchain-demo-index"
        
        if not api_key:
            logger.warning("PINECONE_API_KEY not found. Skipping implementation details.")
            return

        # 1. Initialize Pinecone Client
        pc = Pinecone(api_key=api_key)
        
        # 2. Check or Create Index
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating serverless index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536, # OpenAI embedding size
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        # 3. Use LangChain integration
        embeddings = OpenAIEmbeddings()
        
        # Example: Loading from documents
        docs = [Document(page_content="Pinecone is ideal for cloud-scale RAG.")]
        
        logger.info("Upserting data to Pinecone...")
        vector_store = PineconeVectorStore.from_documents(
            docs, 
            embeddings, 
            index_name=index_name
        )

        # 4. Search
        results = vector_store.similarity_search("Which DB is good for cloud scale?", k=1)
        print(f"Pinecone Result: {results[0].page_content}")

    except Exception as e:
        logger.error(f"Pinecone operation failed: {str(e)}")

if __name__ == "__main__":
    pinecone_demo()
