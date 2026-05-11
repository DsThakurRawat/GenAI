import logging
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MyDatabaseLoader(BaseLoader):
    """
    A production-grade custom loader that simulates loading records from a database
    with robust logging and error handling.
    """

    def __init__(self, table_name: str) -> None:
        self.table_name = table_name
        logger.info(f"Initialized MyDatabaseLoader for table: {table_name}")

    def lazy_load(self) -> Iterator[Document]:
        """
        Implements memory-efficient loading from a data source.
        """
        try:
            logger.info(f"Starting data retrieval from table: {self.table_name}")
            
            # Simulated DB Records (In reality, this would be a DB query)
            records = [
                {"id": 1, "text": "LangChain is great for building agents.", "author": "Alice"},
                {"id": 2, "text": "RAG reduces hallucinations in LLMs.", "author": "Bob"},
                {"id": 3, "text": "Custom loaders are essential for proprietary data.", "author": "Charlie"},
            ]

            if not records:
                logger.warning(f"No records found in table: {self.table_name}")
                return

            for record in records:
                logger.debug(f"Processing record ID: {record['id']}")
                yield Document(
                    page_content=record["text"],
                    metadata={
                        "db_id": record["id"],
                        "author": record["author"],
                        "table": self.table_name,
                        "loader_type": "custom_production"
                    }
                )
            
            logger.info("Retrieval complete.")

        except Exception as e:
            logger.error(f"Critical error during database load: {str(e)}")
            # In production, you might want to retry or raise a custom exception
            raise

if __name__ == "__main__":
    try:
        loader = MyDatabaseLoader(table_name="production_logs")
        for doc in loader.lazy_load():
            print(f"Loaded Doc: {doc.page_content[:50]}...")
    except Exception:
        logger.critical("Process failed due to loader error.")

