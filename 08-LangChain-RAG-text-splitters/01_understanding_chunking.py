import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
### UNDERSTANDING CHUNKING CONCEPTS ###

1. CHUNK SIZE:
   - This is the maximum number of units (characters or tokens) each chunk will contain.
   - INCREASE CHUNK SIZE: 
     - PRO: More context for the LLM. 
     - CON: Higher risk of hitting context window limits, more noise, and slower retrieval.

2. CHUNK OVERLAP:
   - This is the amount of text shared between two adjacent chunks.
   - WHY USE OVERLAP? 
     - To maintain context! If a sentence is split exactly at the boundary, the meaning might be lost. 
     - Overlap ensures that the end of Chunk A and the start of Chunk B share some information.
   - USE CASE: 
     - In complex technical documents or stories where the end of one paragraph relates strongly to the next.

3. EXAMPLE VISUALIZATION:
   Text: "The quick brown fox jumps over the lazy dog."
   Chunk Size: 10, Chunk Overlap: 2
   Chunk 1: "The quick "
   Chunk 2: "k brown fo" (Starts with 'k ' which is the last 2 chars of "The quick ")
"""

def demonstrate_concepts():
    try:
        text = "LangChain provides a variety of text splitters. These are used to break down long documents into smaller pieces."
        chunk_size = 20
        chunk_overlap = 5
        
        logger.info(f"Demonstrating with Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")
        
        # We will use a simple character split for demonstration of the concept
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i:i + chunk_size])
            if i + chunk_size >= len(text):
                break
        
        for idx, chunk in enumerate(chunks):
            print(f"Chunk {idx+1}: '{chunk}'")
            
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")

if __name__ == "__main__":
    demonstrate_concepts()
