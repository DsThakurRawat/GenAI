import logging
import os
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Configure Logging for production quality
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

"""
### EMBEDDINGS GUIDE ###

1. WHAT IS AN EMBEDDING?
   A mathematical vector representing the meaning of text. 
   'Cat' and 'Kitten' will have high cosine similarity.

2. GENERATION TECHNIQUES:
   - API-based: OpenAI (text-embedding-3-small) - Fast, high quality, requires internet.
   - Local-based: HuggingFace (sentence-transformers) - Runs on your CPU/GPU, private, free.
"""

def generate_openai_embeddings(text: str):
    try:
        logger.info("Generating embeddings using OpenAI...")
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Single query embedding
        vector = embeddings_model.embed_query(text)
        
        logger.info(f"Successfully generated vector of size: {len(vector)}")
        print(f"First 5 dimensions: {vector[:5]}")
        return vector
    except Exception as e:
        logger.error(f"OpenAI Embedding failed: {str(e)}")

def generate_huggingface_embeddings(text: str):
    try:
        logger.info("Generating embeddings using Local HuggingFace model...")
        # 'all-MiniLM-L6-v2' is a very popular, lightweight model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        
        vector = embeddings_model.embed_query(text)
        
        logger.info(f"Successfully generated local vector of size: {len(vector)}")
        print(f"First 5 dimensions: {vector[:5]}")
        return vector
    except Exception as e:
        logger.error(f"HuggingFace Embedding failed: {str(e)}")

if __name__ == "__main__":
    sample_text = "Vector stores are the memory of RAG applications."
    
    # generate_openai_embeddings(sample_text) # Uncomment if you have API Key
    generate_huggingface_embeddings(sample_text)
