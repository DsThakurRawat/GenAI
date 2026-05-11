import logging
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
### SEMANTIC CHUNKING (Meaning-Based) ###

1. WHAT IT IS:
   The most advanced way to split text. Instead of counting characters or headers, 
   it uses EMBEDDINGS to measure the similarity between sentences.

2. HOW IT WORKS:
   - It breaks text into sentences.
   - It calculates the embedding (numerical meaning) of each sentence.
   - It groups consecutive sentences that are "semantically similar".
   - If a sentence is very different from the previous ones, it starts a NEW chunk.

3. USE CASE:
   When you want chunks that are purely focused on a single topic, regardless of their length.
   Perfect for high-accuracy RAG.
"""

def semantic_chunking_demo():
    try:
        text = """
        The stock market rallied today after positive inflation data. 
        Investors were optimistic about future interest rate cuts.
        In other news, a new species of frog was discovered in the Amazon rainforest. 
        Biologists say the discovery is significant for biodiversity.
        Back to finance, the Dow Jones reached a new record high.
        """
        
        logger.info("Initializing OpenAIEmbeddings...")
        # Note: Requires OPENAI_API_KEY in environment
        embeddings = OpenAIEmbeddings()
        
        logger.info("Initializing SemanticChunker...")
        # breakpoints: percentile, standard_deviation, interquartile
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        
        logger.info("Splitting text semantically...")
        chunks = splitter.create_documents([text])
        
        logger.info(f"Split into {len(chunks)} semantic chunks.")
        
        for i, chunk in enumerate(chunks):
            print(f"\n--- Semantic Chunk {i+1} ---")
            print(chunk.page_content.strip())
            
    except Exception as e:
        logger.error(f"Semantic chunking failed: {str(e)}")
        logger.info("Tip: Ensure you have an API key and 'langchain-experimental' installed.")

if __name__ == "__main__":
    # semantic_chunking_demo() # Uncomment to run if API key is available
    logger.info("Semantic Chunker script ready. (Requires API Key and experimental package)")
