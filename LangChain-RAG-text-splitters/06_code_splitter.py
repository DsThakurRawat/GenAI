import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
### CODE TEXT SPLITTER ###

1. WHAT IT IS:
   A specialized version of RecursiveCharacterTextSplitter that knows the 
   syntax of programming languages (Python, JS, C++, etc.).

2. WHY USE IT?
   If you use a normal splitter on code, it might break a function in the middle. 
   Code splitters try to keep classes, functions, and loops intact.

3. SUPPORTED LANGUAGES:
   LangChain supports dozens of languages (Python, JavaScript, Java, Go, etc.).
"""

def code_split_demo():
    try:
        python_code = """
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        # This is a very important function
        for item in self.data:
            print(f"Processing {item}")
            
def top_level_function():
    return "Hello World"
        """
        
        logger.info("Initializing RecursiveCharacterTextSplitter for PYTHON...")
        
        # Using from_language helper
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=100,
            chunk_overlap=0
        )
        
        logger.info("Splitting Python code...")
        chunks = splitter.create_documents([python_code])
        
        logger.info(f"Split code into {len(chunks)} chunks.")
        
        for i, chunk in enumerate(chunks):
            print(f"\n--- Code Chunk {i+1} ---")
            print(chunk.page_content)
            
    except Exception as e:
        logger.error(f"Failed to split code: {str(e)}")

if __name__ == "__main__":
    code_split_demo()
