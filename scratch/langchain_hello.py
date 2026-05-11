import os
import logging
from typing import Optional
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import BaseMessage

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GenAI-Learning")

class Config(BaseSettings):
    """Configuration settings using Pydantic Settings."""
    google_api_key: SecretStr = Field(alias="GOOGLE_API_KEY")
    model_name: str = Field(default="gemini-1.5-flash")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

class ChatResponse(BaseModel):
    """Structured response schema."""
    content: str
    tokens: Optional[int] = None

def get_chat_response(messages: list) -> Optional[BaseMessage]:
    """
    Invokes the LLM and handles potential exceptions.
    """
    try:
        config = Config()
        llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key.get_secret_value()
        )
        
        logger.info(f"Invoking model: {config.model_name}")
        response = llm.invoke(messages)
        return response
    except Exception as e:
        logger.error(f"Failed to get response from LLM: {str(e)}", exc_info=True)
        return None

def main():
    logger.info("Starting LangChain Production-Ready Script")
    
    messages = [
        SystemMessage(content="You are a helpful assistant that explains complex concepts simply."),
        HumanMessage(content="What is the significance of the Transformer architecture in GenAI?")
    ]
    
    response = get_chat_response(messages)
    
    if response:
        logger.info("Successfully retrieved response")
        print(f"\n--- Model Response ---\n{response.content}\n")
    else:
        logger.warning("Could not retrieve a valid response.")

if __name__ == "__main__":
    main()
