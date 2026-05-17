import logging
import os
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# =============================================================================
# PRODUCTION LOGGING CONFIGURATION
# =============================================================================
# In production, we use the 'logging' module instead of 'print'. 
# This allows us to track the execution flow, debug errors, and store logs in files.
# format='%(asctime)s - %(levelname)s - %(message)s' ensures every log entry 
# has a timestamp and a clear severity level (INFO, ERROR, etc.).
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file (e.g., OPENAI_API_KEY)
load_dotenv()

"""
###############################################################################
### DEEP DIVE: UNDERSTANDING EMBEDDINGS ###
###############################################################################

1. WHAT IS AN EMBEDDING?
   - At its core, an embedding is a sequence of floating-point numbers (a vector).
   - In Natural Language Processing (NLP), we use models to transform words, 
     sentences, or entire documents into these vectors.
   - The key property: SEMANTIC SIMILARITY translates to GEOMETRIC PROXIMITY.
     Example: The distance between the vectors for 'Apple' and 'iPhone' will be 
     much smaller than the distance between 'Apple' and 'Bicycle'.

2. HOW ARE THEY GENERATED? (THE NLP TECHNIQUE)
   - Modern embeddings use 'Transformers' (like BERT, RoBERTa, or GPT).
   - These models have been trained on massive datasets to understand the context 
     and relationship between words.
   - When we pass text through an embedding model, we usually take the 'hidden state' 
     of the model (the internal numerical representation) as the embedding.

3. TECHNIQUES IN RAG:
   - API-based (OpenAI): Offloads the computation to the cloud. High accuracy, low maintenance.
   - Local-based (HuggingFace): Runs on your own hardware. 100% private, free of cost per token.
###############################################################################
"""

def generate_openai_embeddings(text: str):
    """
    Demonstrates generating high-dimensional vectors using OpenAI's cloud API.
    
    Why OpenAI?
    - State-of-the-art models (text-embedding-3-small/large).
    - Very high dimension count (e.g., 1536) which captures subtle nuances.
    """
    try:
        logger.info(f"Initiating OpenAI embedding generation for text: '{text[:20]}...'")
        
        # 'text-embedding-3-small' is currently the best balance of cost and performance.
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # embed_query() is used to embed a single string (like a user question).
        # embed_documents() would be used for a list of strings (like your knowledge base).
        vector = embeddings_model.embed_query(text)
        
        logger.info(f"Successfully generated OpenAI vector of size: {len(vector)}")
        
        # Printing the start of the vector to show its numerical nature
        print(f"\n--- OpenAI Vector Sample (First 5 dimensions) ---")
        print(vector[:5])
        
        return vector
        
    except Exception as e:
        # Proper exception handling for production: log the error details for debugging.
        logger.error(f"CRITICAL ERROR during OpenAI Embedding generation: {str(e)}")
        raise # Re-raise to alert the calling function

def generate_huggingface_embeddings(text: str):
    """
    Demonstrates generating vectors locally using the HuggingFace library.
    
    Why Local (HuggingFace)?
    - Privacy: Data never leaves your machine.
    - Cost: Completely free to use.
    - Speed: Very fast for small-to-medium datasets on local CPU/GPU.
    """
    try:
        logger.info(f"Initiating Local HuggingFace embedding generation for text: '{text[:20]}...'")
        
        # 'all-MiniLM-L6-v2' is a lightweight model that is extremely popular for RAG.
        # It maps sentences to a 384-dimensional dense vector space.
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        logger.info(f"Loading model: {model_name}...")
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        
        # Generating the vector locally
        vector = embeddings_model.embed_query(text)
        
        logger.info(f"Successfully generated Local vector of size: {len(vector)}")
        
        print(f"\n--- Local HuggingFace Vector Sample (First 5 dimensions) ---")
        print(vector[:5])
        
        return vector
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR during HuggingFace Embedding generation: {str(e)}")
        raise

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    sample_text = "Vector stores allow us to perform semantic search by comparing text embeddings."
    
    print("="*60)
    print("EMBEDDINGS GENERATION DEMONSTRATION")
    print("="*60)
    
    # 1. Local Demonstration (Always works if dependencies are installed)
    generate_huggingface_embeddings(sample_text)
    
    # 2. Cloud Demonstration (Requires OPENAI_API_KEY)
    # if os.getenv("OPENAI_API_KEY"):
    #     generate_openai_embeddings(sample_text)
    # else:
    #     logger.warning("OPENAI_API_KEY not found. Skipping cloud demonstration.")

