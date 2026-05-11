from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.core.logger import logger
from src.core.config import settings

# CONCEPTUAL: Pydantic in GenAI
"""
Pydantic is crucial for 'Structured Output'. 
LLMs usually return strings, but production systems need Objects (JSON).
By defining a Pydantic Model, we can:
1. Validate the LLM response.
2. Get auto-completion in our IDE.
3. Ensure the data fits our database schema.
"""

# 1. Define a Pydantic Schema for the output we want
class AIResponse(BaseModel):
    summary: str = Field(description="A brief 1-sentence summary of the topic")
    facts: list[str] = Field(description="A list of 3 interesting facts")
    sentiment: str = Field(description="The emotional tone of the response (e.g. Positive, Neutral)")

# 2. Setup the Pydantic Output Parser
parser = PydanticOutputParser(pydantic_object=AIResponse)

"""
HUGGINGFACE API MODEL DEMONSTRATION
===================================

This script demonstrates how to use the HuggingFace Inference API to access open-source 
models like Mistral or Llama. It also showcases 'Structured Output' using the 
`PydanticOutputParser`.

### PARAMETER IMPACT GUIDE (Docstring Explanation):

1. **repo_id (str)**:
   - *What it is*: The repository path on HuggingFace Hub.
   - *Variations*:
     - `mistralai/Mistral-7B-Instruct-v0.2`: Excellent at following instructions.
     - `meta-llama/Meta-Llama-3-8B-Instruct`: Balanced performance.
   - *Decision*: Choose based on the task requirements and model availability.

2. **temperature (0.01 to 100.0)**:
   - *Note*: HuggingFace models often fail at exactly 0.0; use 0.01 for deterministic behavior.
   - *Impact*: Higher values increase randomness. 0.5 is usually a safe middle ground.

3. **max_new_tokens (int)**:
   - *Note*: In the Endpoint class, this is often passed in `model_kwargs`.
   - *Impact*: Limits the generated output length.
"""

# Initialize HuggingFace Endpoint
# Using Mistral-7B or Llama-3 via HuggingFace Hub
llm = HuggingFaceEndpoint(
    # PARAMETER: repo_id
    # IMPACT: Selects the specific open-source model hosted on HuggingFace.
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    
    # PARAMETER: huggingfacehub_api_token
    # IMPACT: Authenticates your request to the HF Inference API.
    huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
    
    # PARAMETER: temperature
    # IMPACT: 0.01 for logic/JSON; 0.7+ for creative chat.
    temperature=0.5,
)

# Convert to a Chat Model interface
chat_model = ChatHuggingFace(llm=llm)

def run_huggingface_structured_demo():
    try:
        topic = "The Planet Mars"
        
        # Conceptual: Combining Prompt with Parser Instructions
        # We tell the LLM EXACTLY how to format the output using the parser's instructions
        format_instructions = parser.get_format_instructions()
        
        prompt = f"""
        Explain the following topic: {topic}
        
        {format_instructions}
        """

        logger.info(f"Requesting structured data about {topic} from HuggingFace...")

        # We use .invoke() for structured output (streaming JSON is harder to parse mid-way)
        response = chat_model.invoke([HumanMessage(content=prompt)])
        
        # 3. Parse the raw string into our Pydantic Object
        structured_data = parser.parse(response.content)

        # Now we have a real Python object with type safety!
        logger.success("Successfully parsed structured output.")
        print(f"\n--- Structured Data for {topic} ---")
        print(f"Summary: {structured_data.summary}")
        print(f"Sentiment: {structured_data.sentiment}")
        print("Facts:")
        for fact in structured_data.facts:
            print(f" - {fact}")

    except Exception as e:
        logger.error(f"HuggingFace Error: {e}")
        print("\n[TIP] Ensure HUGGINGFACE_API_KEY is set in your .env")

if __name__ == "__main__":
    run_huggingface_structured_demo()
