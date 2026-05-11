import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_document_with_chunking(file_path: str):
    """
    Demonstrates production-quality document loading and chunking with error handling.
    """
    try:
        logger.info(f"Attempting to load document: {file_path}")
        loader = TextLoader(file_path)
        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} document(s).")

        # 2. Creating a Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True
        )

        logger.info("Starting document splitting into chunks...")
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Document split into {len(chunks)} chunks.")

        if chunks:
            print(f"\n--- Example Chunk ---\n{chunks[0].page_content[:200]}...")
            print(f"Metadata: {chunks[0].metadata}")
        
        return chunks

    except FileNotFoundError:
        logger.error(f"The file '{file_path}' was not found. Please check the path.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    process_document_with_chunking("cricket.txt")

