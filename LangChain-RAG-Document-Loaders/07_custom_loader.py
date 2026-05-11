import logging
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MyCustomTextLoader(BaseLoader):
    """
    A custom loader that reads a file and adds metadata with error handling.
    """

    def __init__(self, file_path: str, custom_tag: str) -> None:
        self.file_path = file_path
        self.custom_tag = custom_tag
        logger.info(f"Initialized MyCustomTextLoader for: {file_path}")

    def lazy_load(self) -> Iterator[Document]:
        """
        Memory-efficient line-by-line file reading.
        """
        try:
            with open(self.file_path, encoding="utf-8") as f:
                logger.info(f"Successfully opened file: {self.file_path}")
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        yield Document(
                            page_content=line,
                            metadata={
                                "source": self.file_path,
                                "line": i,
                                "tag": self.custom_tag
                            }
                        )
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
        except Exception as e:
            logger.error(f"Error reading file {self.file_path}: {str(e)}")

if __name__ == "__main__":
    loader = MyCustomTextLoader(file_path="cricket.txt", custom_tag="sports")
    for doc in loader.lazy_load():
        print(f"Line {doc.metadata['line']}: {doc.page_content[:30]}...")
        if doc.metadata['line'] >= 3: break

