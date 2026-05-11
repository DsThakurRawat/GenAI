import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Temperature controls the "creativity" of the model.
# 0.0 = Deterministic (best for facts/logic)
# 0.7 - 1.0 = Creative (standard for chat)
# 1.5+ = Very random (can lead to hallucinations or nonsense)

def run_temperature_demo():
    logger.info("Initializing model with high temperature...")
    try:
        # Note: Different models have different max temperatures. 
        # Gemini usually supports up to 2.0.
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=1.2)
        
        prompt = "Write a 5 line poem on cricket"
        logger.info(f"Invoking model with prompt: {prompt}")
        
        result = model.invoke(prompt)
        
        print("--- AI Response (Creative) ---")
        print(result.content)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_temperature_demo()