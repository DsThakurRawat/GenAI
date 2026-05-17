"""
Demonstration of LangChain's with_structured_output using TypedDict and Annotations.
"""
import logging
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

logging.basicConfig(level=logging.INFO, format='\n%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize the Gemini model. 
# We use temperature 0.0 because structured data extraction should be deterministic.
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# --- 1. SCHEMAS, INHERITANCE & ANNOTATIONS ---

# We inherit from TypedDict to create our schema blueprint.
class Review(TypedDict):
    """
    This schema dictates exactly how the LLM must respond.
    
    What is 'Annotated'?
    Annotated[Type, Metadata] allows us to attach descriptions to our types.
    The LLM doesn't just see "list[str]". LangChain reads the string description
    and passes it to the LLM so it knows *what* the list should contain.
    """
    
    # Annotated[TYPE, "DESCRIPTION FOR THE LLM"]
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    
    # Literal means the string MUST be exactly one of the provided options.
    sentiment: Annotated[Literal["pos", "neg", "neutral"], "Return sentiment of the review either negative, positive or neutral"]
    
    # Optional means this field might not exist if the LLM can't find the info.
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

def run_structured_extraction():
    # --- 2. WITH_STRUCTURED_OUTPUT ---
    # This is the magic LangChain method. 
    # Under the hood, it converts our 'Review' TypedDict and its Annotations into a JSON Schema.
    # It then forces the LLM to reply using exactly that JSON Schema.
    structured_model = model.with_structured_output(Review)
    
    review_text = """
    I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! 
    The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, 
    or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
    
    The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. 
    What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. 
    Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
    
    However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes 
    with bloatware—why do I need five different Samsung apps for things Google already provides? 
    The $1,300 price tag is also a hard pill to swallow.
    
    Review by Nitish Singh
    """
    
    logger.info("Sending review to LLM to extract structured data...")
    
    # The return value will be a Python dictionary perfectly matching our TypedDict!
    result = structured_model.invoke(review_text)
    
    logger.info("--- Extracted Data ---")
    logger.info(f"Reviewer Name: {result.get('name')}")
    logger.info(f"Sentiment: {result.get('sentiment')}")
    logger.info(f"Summary: {result.get('summary')}")
    
    # Printing the whole dictionary beautifully
    logger.info("\nFull Dictionary Output:")
    logger.info(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_structured_extraction()