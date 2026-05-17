import logging
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def run_prompt_template_demo():
    # Modern recommendation: use ChatPromptTemplate for chat models, 
    # but PromptTemplate is still useful for raw strings or legacy LLMs.
    
    # Initialization using from_template is the preferred shorthand
    template = PromptTemplate.from_template(
        "Greet this person in 5 languages. The name of the person is {name}"
    )

    logger.info("Initializing model...")
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        
        # Fill the values of the placeholders
        formatted_prompt = template.invoke({'name': 'nitish'})
        logger.info(f"Formatted prompt: {formatted_prompt.text}")

        # Invoke the model
        result = model.invoke(formatted_prompt)
        
        print("--- AI Response ---")
        print(result.content)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_prompt_template_demo()
