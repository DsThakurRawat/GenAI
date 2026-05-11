import logging
from pydantic import BaseModel, Field, validator
from langchain_core.prompts import ChatPromptTemplate
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperSummaryInput(BaseModel):
    paper_input: str = Field(..., min_length=1, description="The title of the research paper.")
    style_input: str = Field(default="Academic", description="The style of the explanation.")
    length_input: str = Field(default="Medium", description="The length of the explanation.")

    @validator('style_input')
    def validate_style(cls, v):
        allowed_styles = ["Academic", "Simple", "Technical", "Layman"]
        if v not in allowed_styles:
            logger.warning(f"Style '{v}' not in recommended list {allowed_styles}. Proceeding anyway.")
        return v

def generate_prompt_template():
    # Modern ChatPromptTemplate is preferred for chat models
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant specializing in clear summaries."),
        ("human", """
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""")
    ])

    # Save to JSON (using the internal representation of the template)
    try:
        # ChatPromptTemplate can be serialized to a dictionary then JSON
        with open('template.json', 'w') as f:
            json.dump(template.to_json(), f, indent=2)
        logger.info("Successfully saved template.json")
    except Exception as e:
        logger.error(f"Failed to save template: {e}")

    return template

if __name__ == "__main__":
    # Example validation using Pydantic
    try:
        user_input = PaperSummaryInput(
            paper_input="Attention Is All You Need",
            style_input="Simple",
            length_input="Short"
        )
        logger.info(f"Validated input for paper: {user_input.paper_input}")
        
        chat_template = generate_prompt_template()
        
        # Example of formatting the prompt
        formatted_messages = chat_template.format_messages(**user_input.dict())
        logger.info(f"Generated {len(formatted_messages)} messages for the chat model.")
        
    except Exception as e:
        logger.critical(f"Validation error: {e}")