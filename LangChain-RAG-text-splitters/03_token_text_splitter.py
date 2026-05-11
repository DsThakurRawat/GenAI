import logging
from langchain_text_splitters import TokenTextSplitter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
### TOKEN TEXT SPLITTER ###

1. WHAT IT IS:
   LLMs don't see characters; they see TOKENS. A TokenTextSplitter splits text 
   based on the number of tokens, ensuring you never exceed an LLM's context limit.

2. WHY USE IT?
   If you set chunk_size=500 in CharacterTextSplitter, that's 500 characters. 
   But an LLM's limit is in tokens (e.g., 4096 tokens). 
   Token splitting is more precise for LLM budget management.

3. HOW IT WORKS:
   It uses a tokenizer (like tiktoken for OpenAI) to convert text to tokens, 
   then splits based on the token count.
"""

def token_split_demo():
    try:
        text = "LangChain is a framework for developing applications powered by language models." * 10
        
        logger.info("Initializing TokenTextSplitter (encoding: cl100k_base)...")
        
        # This uses the tiktoken library under the hood
        splitter = TokenTextSplitter(
            chunk_size=20,
            chunk_overlap=5
        )
        
        logger.info("Splitting text by tokens...")
        chunks = splitter.split_text(text)
        
        logger.info(f"Created {len(chunks)} token-based chunks.")
        
        for i, chunk in enumerate(chunks[:3]): # Show first 3
            print(f"\n--- Token Chunk {i+1} ---")
            print(chunk)
            
    except ImportError:
        logger.error("The 'tiktoken' library is required for TokenTextSplitter. Install it with 'pip install tiktoken'.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    token_split_demo()
