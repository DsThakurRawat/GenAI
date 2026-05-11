from src.core.logger import logger
from src.core.config import settings

def main():
    logger.info(f"Starting {settings.APP_NAME}...")
    
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not found in environment. Some features may not work.")
    else:
        logger.success("OpenAI API Key loaded successfully.")

    # Application logic goes here
    logger.debug("Debug mode is active.")
    logger.info("Application setup complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Application crashed: {e}")
