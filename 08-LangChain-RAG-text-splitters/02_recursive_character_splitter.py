import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
### RECURSIVE CHARACTER TEXT SPLITTER ###

1. WHAT IT IS:
   The recommended splitter for generic text. It tries to split on a list of characters 
   in order until the chunks are small enough.
   Default list: ["\n\n", "\n", " ", ""]

2. WHY IT IS BETTER:
   Unlike CharacterTextSplitter, it tries to keep paragraphs (\n\n), then sentences (\n), 
   then words ( ) together as much as possible. This preserves the SEMANTIC STRUCTURE of the text.

3. PARAMETERS:
   - chunk_size: Max characters per chunk.
   - chunk_overlap: Overlap between chunks.
   - separators: Custom list of characters to split on.
"""

def recursive_split_demo():
    try:
        # Sample long text
        text = """
        LangChain is a framework for developing applications powered by language models.
        It enables applications that are:
        1. Data-aware: connect a language model to other sources of data.
        2. Agentic: allow a language model to interact with its environment.
        
        The main value props of LangChain are:
        - Components: modular abstractions for the components necessary to work with LLMs.
        - Off-the-shelf chains: a structured assembly of components for accomplishing a specific higher-level task.
        """
        
        logger.info("Initializing RecursiveCharacterTextSplitter...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        
        logger.info("Splitting text...")
        chunks = splitter.create_documents([text])
        
        logger.info(f"Split successful! Created {len(chunks)} chunks.")
        
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk.page_content)
            
    except Exception as e:
        logger.error(f"Failed to split text: {str(e)}")

if __name__ == "__main__":
    recursive_split_demo()
