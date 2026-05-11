import logging
from langchain_community.document_loaders import WebBaseLoader

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scrape_url_production(url: str):
    """
    Production-ready web scraping with error handling for network and parsing issues.
    """
    try:
        logger.info(f"Sending request to scrape URL: {url}")
        loader = WebBaseLoader(url)
        
        # Some websites block basic scrapers; WebBaseLoader uses requests/BeautifulSoup
        docs = loader.load()
        
        if not docs:
            logger.warning("No content was retrieved from the URL.")
            return None

        logger.info(f"Successfully scraped content from: {url}")
        print(f"\nTitle: {docs[0].metadata.get('title')}")
        print(f"Snippet: {docs[0].page_content.strip()[:200]}...")
        
        return docs

    except Exception as e:
        logger.error(f"An error occurred while scraping {url}: {str(e)}")
        # In a real app, you might try a proxy or a more advanced loader here

if __name__ == "__main__":
    target_url = "https://python.langchain.com/v0.2/docs/how_to/#document-loaders"
    scrape_url_production(target_url)

