import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
###############################################################################
### ADVANCED: PARENT DOCUMENT RETRIEVER ###
###############################################################################

1. THE DILEMMA:
   - Small chunks (e.g., 200 chars) are better for retrieval accuracy (finding the needle).
   - Large context (e.g., 2000 chars) is better for the LLM to generate a high-quality answer.

2. THE SOLUTION:
   - Split parents into children.
   - Search on the children.
   - Return the parent.

3. STRUCTURE:
   - Vector Store: Stores Child Chunks (Embeddings).
   - Document Store (Docstore): Stores Parent Documents (Full Text).
###############################################################################
"""

def parent_doc_demo():
    try:
        embeddings = OpenAIEmbeddings()
        
        # 1. Splitters
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

        # 2. Storage
        vectorstore = Chroma(collection_name="full_documents", embedding_function=embeddings)
        store = InMemoryStore() # Local storage for parents

        # 3. Create Retriever
        logger.info("Initializing ParentDocumentRetriever...")
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        # 4. Add Documents
        logger.info("Adding documents...")
        docs = [
            Document(page_content="""
                The Industrial Revolution began in Great Britain. 
                It was a period of global transition to new manufacturing processes.
                (Imagine another 1000 words about steam engines and textile factories here.)
                One major invention was the Watt steam engine.
            """, metadata={"source": "history_book"})
        ]
        retriever.add_documents(docs)

        # 5. Search
        # This will find the child chunk mentioning 'steam engine' 
        # but will return the entire 1000-word parent.
        query = "What was a major invention during the Industrial Revolution?"
        logger.info(f"Query: {query}")
        
        results = retriever.invoke(query)

        print("\n" + "="*30)
        print("PARENT DOCUMENT RESULTS")
        print("="*30)
        print(f"Number of parents retrieved: {len(results)}")
        print(f"Parent Content Sample: {results[0].page_content[:200]}...")

    except Exception as e:
        logger.error(f"Parent document retrieval failed: {str(e)}")

if __name__ == "__main__":
    parent_doc_demo()
