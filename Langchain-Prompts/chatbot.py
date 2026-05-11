import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_community.chat_message_histories import InMemoryChatMessageHistory
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class ChatBotConfig(BaseModel):
    model_name: str = Field(default="gemini-1.5-flash")
    max_history_tokens: int = Field(default=2000)
    system_prompt: str = Field(default="You are a helpful AI assistant.")

def run_chatbot():
    config = ChatBotConfig()
    
    try:
        model = ChatGoogleGenerativeAI(model=config.model_name)
        logger.info(f"Chatbot started with model: {config.model_name}")
    except Exception as e:
        logger.error(f"Failed to start chatbot: {e}")
        return

    # Modern way to store history
    history = InMemoryChatMessageHistory()
    history.add_message(SystemMessage(content=config.system_prompt))

    print("--- AI Chatbot (type 'exit' to quit) ---")
    
    while True:
        try:
            user_input = input('You: ').strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                logger.info("Exiting chatbot.")
                break
            
            history.add_user_message(user_input)
            
            # Handling growing history: Trim messages to stay within token limits
            # This ensures the LLM doesn't get overwhelmed and stays cost-effective
            trimmed_history = trim_messages(
                history.messages,
                max_tokens=config.max_history_tokens,
                strategy="last",
                token_counter=model,
                include_system=True, # Always keep the system prompt
            )
            
            logger.info(f"Invoking model with {len(trimmed_history)} messages (trimmed).")
            result = model.invoke(trimmed_history)
            
            history.add_ai_message(result.content)
            print(f"AI: {result.content}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    logger.info("Chat history saved in memory.")

if __name__ == "__main__":
    run_chatbot()