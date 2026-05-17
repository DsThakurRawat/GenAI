import logging
from langchain_community.retrievers import WikipediaRetriever, ArxivRetriever

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
###############################################################################
### EXTERNAL SOURCE RETRIEVERS ###
###############################################################################

RAG doesn't always have to use a Vector Store. You can retrieve data 
directly from the internet or existing APIs!

1. WIKIPEDIA RETRIEVER:
   - Fetches summaries directly from Wikipedia.
   - Great for general knowledge questions.

2. ARXIV RETRIEVER:
   - Fetches scientific papers from the Arxiv repository.
   - Great for technical and research-oriented RAG.
###############################################################################
"""

def external_retrievers_demo():
    try:
        # 1. Wikipedia Demo
        logger.info("Initializing WikipediaRetriever...")
        wiki_retriever = WikipediaRetriever()
        
        wiki_results = wiki_retriever.invoke("Quantum Computing")
        
        print("\n" + "="*30)
        print("WIKIPEDIA RESULT")
        print("="*30)
        print(f"Title: {wiki_results[0].metadata['title']}")
        print(f"Summary: {wiki_results[0].page_content[:300]}...")

        # 2. Arxiv Demo
        logger.info("Initializing ArxivRetriever...")
        arxiv_retriever = ArxivRetriever(load_max_docs=1)
        
        arxiv_results = arxiv_retriever.invoke("Attention is All You Need")
        
        print("\n" + "="*30)
        print("ARXIV RESULT")
        print("="*30)
        print(f"Title: {arxiv_results[0].metadata['Title']}")
        print(f"Summary: {arxiv_results[0].page_content[:300]}...")

    except Exception as e:
        logger.error(f"External retrieval failed: {str(e)}")

if __name__ == "__main__":
    external_retrievers_demo()
