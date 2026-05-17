import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ==========================================
# 1. THE BASICS: PromptTemplate
# ==========================================
def basic_prompt_template():
    logger.info("--- 1. Basic PromptTemplate ---")
    
    template = "Explain the concept of {topic} in the style of {style}."
    prompt = PromptTemplate.from_template(template)
    
    formatted_prompt = prompt.format(topic="Recursion", style="a pirate")
    
    print(f"Template: {template}")
    print(f"Formatted: {formatted_prompt}")


# ==========================================
# 2. THE MODERN WAY: ChatPromptTemplate
# ==========================================
def chat_prompt_template_demo():
    logger.info("--- 2. ChatPromptTemplate ---")
    
    # Modern approach using tuples (role, content)
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a professional {role} assistant."),
        ("human", "How do I handle {issue}?"),
    ])
    
    formatted_messages = chat_template.format_messages(
        role="legal", 
        issue="a noise complaint from a neighbor"
    )
    
    print("Formatted Messages:")
    for msg in formatted_messages:
        print(f"[{type(msg).__name__}]: {msg.content}")


# ==========================================
# 3. MESSAGE TYPES (Explicitly)
# ==========================================
def explicit_messages_demo():
    logger.info("--- 3. Explicit Message Types ---")
    
    messages: List[BaseMessage] = [
        SystemMessage(content="You are a helpful travel agent."),
        HumanMessage(content="I want to visit Tokyo."),
        AIMessage(content="Tokyo is amazing! Would you like to see a 3-day itinerary?"),
        HumanMessage(content="Yes, please focus on food.")
    ]
    
    print(f"Sequence length: {len(messages)} messages")
    print(f"Last message type: {type(messages[-1]).__name__}")


# ==========================================
# 4. PARTIAL PROMPTS
# ==========================================
def partial_prompts_demo():
    logger.info("--- 4. Partial Prompts ---")
    
    base_prompt = PromptTemplate.from_template("{greeting}, I am {name}. How can I help you?")
    
    # Pre-filling some variables
    partial_prompt = base_prompt.partial(greeting="Hello")
    
    # Only 'name' is needed now
    final_output = partial_prompt.format(name="Antigravity")
    
    print(f"Final Prompt: {final_output}")


# ==========================================
# 5. PYDANTIC INTEGRATION (Modern Practice)
# ==========================================
class LessonConfig(BaseModel):
    topic: str = Field(..., description="The lesson topic")
    difficulty: str = Field(default="Beginner")
    student_name: Optional[str] = None

def pydantic_integration_demo():
    logger.info("--- 5. Pydantic Integration ---")
    
    config = LessonConfig(topic="LangChain LCEL", student_name="Divyansh")
    
    template = ChatPromptTemplate.from_template(
        "Hi {student_name}, today we will learn about {topic} at a {difficulty} level."
    )
    
    # We can pass the pydantic model as a dictionary
    formatted = template.format(**config.dict())
    print(f"Formatted with Pydantic: {formatted}")


# ==========================================
# 6. MESSAGE PLACEHOLDERS
# ==========================================
def message_placeholder_demo():
    logger.info("--- 6. MessagesPlaceholder ---")
    
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    history = [
        HumanMessage(content="Hi, I'm Divyansh."),
        AIMessage(content="Nice to meet you, Divyansh! How can I help you today?")
    ]
    
    formatted = chat_template.format_messages(
        history=history,
        input="What is my name?"
    )
    
    print("Formatted with History:")
    for msg in formatted:
        print(f"[{type(msg).__name__}]: {msg.content}")


if __name__ == "__main__":
    logger.info("Starting LangChain Prompts Lesson...")
    
    basic_prompt_template()
    chat_prompt_template_demo()
    explicit_messages_demo()
    partial_prompts_demo()
    pydantic_integration_demo()
    message_placeholder_demo()
    
    logger.info("Lesson Complete!")
