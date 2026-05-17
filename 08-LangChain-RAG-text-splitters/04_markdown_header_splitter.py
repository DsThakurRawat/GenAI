import logging
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
### MARKDOWN HEADER TEXT SPLITTER (Structure-Based) ###

1. WHAT IT IS:
   Instead of splitting by character count, this splitter understands Markdown headers (#, ##, ###).
   It splits the text based on where the headers are.

2. WHY USE IT? (The 'Structure-Based' Advantage)
   It preserves the context of sections! If you have a section called "Installation", 
   all text under that header is kept together in a way that the metadata knows it's about Installation.

3. METADATA ENRICHMENT:
   This splitter automatically adds the header names to the chunk's metadata, 
   which is incredibly useful for RAG retrieval.
"""

def markdown_split_demo():
    try:
        markdown_text = """
# LangChain Documentation

## Introduction
LangChain is a framework for language models.

## Installation
You can install it via pip:
`pip install langchain`

### Requirements
- Python 3.8+
- OpenAI API Key

## Usage
Import the library and start building.
        """
        
        # Define the headers to split on and their names in metadata
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        logger.info("Initializing MarkdownHeaderTextSplitter...")
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        logger.info("Splitting Markdown text...")
        chunks = splitter.split_text(markdown_text)
        
        logger.info(f"Successfully split into {len(chunks)} chunks based on headers.")
        
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Metadata: {chunk.metadata}")
            print(f"Content: {chunk.page_content.strip()}")
            
    except Exception as e:
        logger.error(f"Error during Markdown splitting: {str(e)}")

if __name__ == "__main__":
    markdown_split_demo()
