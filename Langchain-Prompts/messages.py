import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ChatConfig(BaseModel):
    model_name: str = Field(default="gemini-1.5-flash")
    temperature: float = Field(default=0.7)

def run_chat_example():
    config = ChatConfig()
    
    # Initialize the model
    # Note: Ensure GOOGLE_API_KEY is in your .env file
    try:
        model = ChatGoogleGenerativeAI(model=config.model_name, temperature=config.temperature)
        logger.info(f"Initialized model: {config.model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return

    # Using LangChain message classes for structure
    messages: List[BaseMessage] = [
        SystemMessage(content='You are a helpful assistant specialized in LangChain.'),
        HumanMessage(content='Explain how AIMessage and HumanMessage differ from raw dictionaries.')
    ]

    logger.info("Sending messages to model...")
    try:
        result = model.invoke(messages)
        
        # Append the AI's response as an AIMessage
        messages.append(result)
        
        logger.info("Received response from AI.")
        for msg in messages:
            role = type(msg).__name__
            print(f"[{role}]: {msg.content}")
            
    except Exception as e:
        logger.error(f"Error during model invocation: {e}")

if __name__ == "__main__":
    run_chat_example()

