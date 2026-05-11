import logging
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Using a list of tuples is the modern way to define ChatPromptTemplates
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

logger.info("Invoking chat template...")
try:
    prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'Dusra'})
    
    print("--- Generated Chat Prompt ---")
    print(prompt)
    
    # Accessing messages individually
    for msg in prompt.to_messages():
        print(f"[{type(msg).__name__}]: {msg.content}")

except Exception as e:
    logger.error(f"Error: {e}")