import logging
import os
from langchain_community.document_loaders import CSVLoader

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_csv_production(file_path: str, source_col: str):
    """
    Robust CSV loading for production datasets.
    """
    try:
        logger.info(f"Loading CSV file: {file_path} with source column: {source_col}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found at {file_path}")

        loader = CSVLoader(file_path=file_path, source_column=source_col)
        
        # In production, we iterate lazily for large files
        docs_loaded = 0
        for row in loader.lazy_load():
            docs_loaded += 1
            if docs_loaded == 1:
                logger.info("First row loaded successfully.")
                print(f"\n--- Sample Row Content ---\n{row.page_content}")
        
        logger.info(f"Total rows processed: {docs_loaded}")

    except KeyError:
        logger.error(f"Error: The source column '{source_col}' was not found in the CSV.")
    except Exception as e:
        logger.error(f"Unexpected error loading CSV: {str(e)}")

if __name__ == "__main__":
    # Note: Ensure the CSV actually has 'User ID' or change to a column that exists
    load_csv_production("Social_Network_Ads.csv", source_col="User ID")

