import logging
from langchain_text_splitters import HTMLHeaderTextSplitter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
### HTML HEADER TEXT SPLITTER ###

1. WHAT IT IS:
   Similar to the Markdown splitter, but for HTML. It understands <h1>, <h2>, etc.

2. WHY USE IT?
   When you scrape a website using WebBaseLoader, you get raw HTML or text. 
   Using this splitter allows you to keep sections together based on the website's structure.

3. METADATA:
   It adds the HTML header hierarchy to the metadata, making the chunks much easier to search.
"""

def html_split_demo():
    try:
        html_string = """
        <!DOCTYPE html>
        <html>
        <body>
            <div>
                <h1>Main Topic: AI</h1>
                <p>Some intro text about AI.</p>
                <div>
                    <h2>Subtopic: Machine Learning</h2>
                    <p>Details about ML.</p>
                </div>
                <h2>Subtopic: Deep Learning</h2>
                <p>Details about DL.</p>
            </div>
        </body>
        </html>
        """
        
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
        ]
        
        logger.info("Initializing HTMLHeaderTextSplitter...")
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        logger.info("Splitting HTML content...")
        chunks = splitter.split_text(html_string)
        
        logger.info(f"Created {len(chunks)} chunks from HTML.")
        
        for i, chunk in enumerate(chunks):
            print(f"\n--- HTML Chunk {i+1} ---")
            print(f"Metadata: {chunk.metadata}")
            print(f"Content: {chunk.page_content.strip()}")
            
    except Exception as e:
        logger.error(f"HTML splitting failed: {str(e)}")

if __name__ == "__main__":
    html_split_demo()
