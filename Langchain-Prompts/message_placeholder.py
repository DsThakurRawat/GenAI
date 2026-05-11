import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Define the template
# MessagesPlaceholder is where the list of previous messages will be injected
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# Correct way to load history: 
# MessagesPlaceholder expects a list of BaseMessage objects (HumanMessage, AIMessage, etc.)
chat_history = []

try:
    with open('chat_history.txt', 'r') as f:
        # Example logic: alternating Human and AI messages from file lines
        for i, line in enumerate(f):
            content = line.strip()
            if not content: continue
            
            if i % 2 == 0:
                chat_history.append(HumanMessage(content=content))
            else:
                chat_history.append(AIMessage(content=content))
    logger.info(f"Loaded {len(chat_history)} messages from history file.")
except FileNotFoundError:
    logger.warning("chat_history.txt not found. Starting with empty history.")

# Create the prompt
# We pass the list of message objects to the 'chat_history' variable
try:
    prompt = chat_template.invoke({
        'chat_history': chat_history, 
        'query': 'Where is my refund?'
    })
    
    logger.info("Successfully generated prompt with history.")
    print("--- Generated Prompt Messages ---")
    for msg in prompt.to_messages():
        print(f"[{type(msg).__name__}]: {msg.content}")

except Exception as e:
    logger.error(f"Error generating prompt: {e}")