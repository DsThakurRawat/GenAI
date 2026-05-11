import logging
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def answer_page_question_production(url: str, user_query: str):
    """
    Production-quality backend logic for a Chrome Extension.
    Includes scraping, context management, and LLM orchestration with error handling.
    """
    try:
        logger.info(f"Received request for URL: {url}")
        
        # 1. Scrape the page
        logger.info("Initializing WebBaseLoader...")
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        if not docs or not docs[0].page_content:
            logger.error("Failed to extract content from the page.")
            return "Sorry, I couldn't read the content of this page."
            
        page_text = docs[0].page_content
        logger.info(f"Extracted {len(page_text)} characters from page.")

        # 2. Setup LLM and Prompt
        # In production, use environment variables for API keys
        logger.info("Orchestrating LLM chain...")
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful Chrome Extension Assistant. 
        Context from the page:
        {context}
        
        Question: {question}
        """)
        
        chain = prompt | llm | StrOutputParser()
        
        # 3. Execution
        # Note: Truncation is a basic approach; RAG (chunking + retrieval) is better for long pages.
        response = chain.invoke({
            "context": page_text[:12000], 
            "question": user_query
        })
        
        logger.info("Successfully generated response.")
        return response

    except Exception as e:
        logger.error(f"Error in backend processing: {str(e)}")
        return "An internal error occurred while processing your request."

if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    test_query = "Summarize the key milestones of AI."
    # response = answer_page_question_production(test_url, test_query)
    # print(f"Response: {response}")
    logger.info("Production backend logic scaffold ready.")

