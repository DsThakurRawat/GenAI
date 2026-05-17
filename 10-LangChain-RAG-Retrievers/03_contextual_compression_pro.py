import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
###############################################################################
### ADVANCED: CONTEXTUAL COMPRESSION ###
###############################################################################

1. THE PROBLEM (Noise):
   - A traditional retriever returns entire chunks (e.g., 500 words).
   - Often, only 1 sentence in that chunk is relevant to the question.
   - Sending the whole chunk to the LLM increases token cost and can 
     dilute the answer with "irrelevant noise".

2. THE SOLUTION:
   - A ContextualCompressionRetriever uses a "Compressor" to post-process 
     the retrieved documents.
   - It extracts ONLY the snippets relevant to the specific query.

3. WORKFLOW:
   User Query -> Base Retriever -> Documents -> Compressor -> COMPRESSED DOCS -> LLM
###############################################################################
"""

def contextual_compression_demo():
    try:
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # 1. Setup sample data (Long documents with mixed info)
        docs = [
            Document(page_content="""
                The Grand Canyon is a famous natural site in Arizona. 
                Photosynthesis is the process by which plants convert light into energy. 
                Many tourists visit the Grand Canyon every year to see its red rocks.
            """),
            Document(page_content="""
                Einstein proposed the theory of relativity in 1905. 
                Gravity is the force that keeps planets in orbit around the sun.
                Relativity changed how we understand time and space.
            """)
        ]
        vector_store = Chroma.from_documents(docs, embeddings)

        # 2. Setup the Compressor
        # LLMChainExtractor will use the LLM to extract ONLY the relevant part of the doc.
        logger.info("Initializing LLMChainExtractor (Compressor)...")
        compressor = LLMChainExtractor.from_llm(llm)

        # 3. Create the Compression Retriever
        logger.info("Creating ContextualCompressionRetriever...")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vector_store.as_retriever()
        )

        # 4. Search
        query = "What is photosynthesis?"
        logger.info(f"Query: {query}")
        
        compressed_docs = compression_retriever.invoke(query)

        print("\n" + "="*30)
        print("COMPRESSED RESULTS")
        print("="*30)
        for i, doc in enumerate(compressed_docs):
            print(f"Compressed Doc {i+1}: '{doc.page_content.strip()}'")
            print("-" * 20)

    except Exception as e:
        logger.error(f"Compression failed: {str(e)}")

if __name__ == "__main__":
    contextual_compression_demo()
