import logging
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_directory_production(path: str, glob_pattern: str):
    """
    Production-quality directory loading with lazy_load and error handling.
    """
    try:
        logger.info(f"Scanning directory: {path} for pattern: {glob_pattern}")
        
        if not os.path.isdir(path):
            logger.error(f"Directory path '{path}' does not exist.")
            return

        loader = DirectoryLoader(
            path=path,
            glob=glob_pattern,
            loader_cls=TextLoader
        )

        logger.info("Beginning lazy load of directory contents...")
        count = 0
        for doc in loader.lazy_load():
            count += 1
            logger.debug(f"Loaded: {doc.metadata.get('source')}")
            if count == 1:
                print(f"\n--- First File Sample ---\n{doc.page_content[:100]}...")
        
        logger.info(f"Successfully processed {count} files from the directory.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during directory processing: {str(e)}")

if __name__ == "__main__":
    load_directory_production("./", "*.txt")

