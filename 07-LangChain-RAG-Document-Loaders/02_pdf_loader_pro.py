import logging
from langchain_community.document_loaders import PyPDFLoader

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pdf_production(file_path: str):
    """
    Robust PDF loading with error handling and logging.
    """
    try:
        logger.info(f"Starting PDF load for: {file_path}")
        loader = PyPDFLoader(file_path)
        
        # We use lazy_load here as it's better for production memory management
        logger.info("Iterating through pages using lazy_load...")
        pages_count = 0
        for page in loader.lazy_load():
            pages_count += 1
            if pages_count == 1:
                print(f"\n--- First Page Content Snippet ---\n{page.page_content[:200]}...")
                print(f"Metadata: {page.metadata}")
        
        logger.info(f"Successfully processed {pages_count} pages.")

    except FileNotFoundError:
        logger.error(f"Error: The PDF file '{file_path}' was not found.")
    except Exception as e:
        logger.error(f"Failed to load PDF. Error details: {str(e)}")

if __name__ == "__main__":
    load_pdf_production("dl-curriculum.pdf")

