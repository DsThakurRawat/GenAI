from langchain_openai import ChatOpenAI
from src.core.logger import logger
from src.core.config import settings
import sys

"""
OPENAI CHAT MODEL DEMONSTRATION
==============================

This script demonstrates how to initialize and use OpenAI's Chat Models (GPT) 
with detailed control over generation parameters.

### PARAMETER IMPACT GUIDE (Docstring Explanation):

1. **model**: 
   - `gpt-4o`: Omni model. High intelligence, high speed, and multimodal support.
   - `gpt-3.5-turbo`: Legacy model. Fast and cheap, but less reliable for complex logic.

2. **temperature (0.0 to 2.0)**:
   - Controls randomness. 
   - **Variation**: 0.0 results in consistent, factual outputs. 1.0+ results in diverse, creative outputs.

3. **top_p (0.0 to 1.0)**:
   - Nucleus sampling. Limits the model's vocabulary choice to the most likely tokens.
   - **Variation**: 0.1 means the model only considers the top 10% of likely words.

4. **frequency_penalty (-2.0 to 2.0)**:
   - Penalizes tokens based on how many times they have already appeared.
   - **Impact**: Positive values prevent the model from repeating the exact same phrases.

5. **presence_penalty (-2.0 to 2.0)**:
   - Penalizes tokens based on whether they have appeared at all.
   - **Impact**: Positive values encourage the model to branch out into new topics.
"""

# Production Setup for ChatOpenAI
# Using settings ensures we don't leak API keys and can change models easily
chat_model = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    
    # PARAMETER: model
    # IMPACT: Determines intelligence and cost. gpt-4o is the current gold standard.
    model="gpt-4o", 
    
    # PARAMETER: temperature
    # IMPACT: 0.0 for accuracy; 0.7 for conversational feel; 1.0 for creativity.
    temperature=0.5,    
    
    # PARAMETER: max_tokens
    # IMPACT: Limits response length to avoid high costs or rambling.
    max_tokens=1000,    
    
    # PARAMETER: top_p
    # IMPACT: Lower values (e.g., 0.2) make the model more confident/less varied.
    top_p=0.9,          
    
    # PARAMETER: frequency_penalty
    # IMPACT: Set > 0 to stop the model from saying the same thing over and over.
    frequency_penalty=0.5, 
    
    # PARAMETER: presence_penalty
    # IMPACT: Set > 0 to make the model more likely to talk about new things.
    presence_penalty=0.5,  
    
    # PARAMETER: streaming
    # IMPACT: If True, tokens are yielded as they are generated for a "live" feel.
    streaming=True      
)

#what is streaming in chatbot ?
"""
STREAMING vs BATCH PROCESSING
-----------------------------
1. Streaming (.stream()):
   - "Typewriter" effect: Chunks appear as they are generated.
   - Faster PERCEIVED response time (Time To First Token is low).
   - Essential for UI/UX in chatbots (e.g., ChatGPT).
   - Technical: Uses Server-Sent Events (SSE) or WebSockets to push tokens.

2. Batch Processing (.invoke() or .batch()):
   - Wait for the entire response to be finished before seeing anything.
   - Faster RAW processing (less network overhead per token).
   - Technical: Standard HTTP Request-Response.
"""

#practical use case of batch processing 
"""
USE CASES FOR BATCH PROCESSING:
1. Background Tasks: Generating summaries for 1000 PDFs at once.
2. Analytics/ETL: Converting unstructured data to JSON in a pipeline.
3. Evaluation: Running test cases through an LLM to check accuracy.
4. Cost Efficiency: In some providers, batch APIs are 50% cheaper (e.g., OpenAI Batch API).
"""

from pydantic import BaseModel, Field

# CONCEPTUAL: Pydantic & Structured Output
# In production, we use .with_structured_output() to force the LLM 
# to return a valid Pydantic object instead of a string.
class Translation(BaseModel):
    original_text: str = Field(description="The input text")
    translated_text: str = Field(description="The text translated to French")
    confidence_score: float = Field(description="Score between 0 and 1")

# Create a structured version of our model
structured_llm = chat_model.with_structured_output(Translation)

def run_chat_demo():
    try:
        query = "Hello, how are you? I am learning LangChain today."
        logger.info(f"Sending Query for Structured Translation: {query}")

        # This returns a Translation OBJECT, not a string!
        result = structured_llm.invoke(query)
        
        logger.success("Structured output received.")
        print(f"\nOriginal: {result.original_text}")
        print(f"French: {result.translated_text}")
        print(f"Confidence: {result.confidence_score}")

    except Exception as e:
        logger.error(f"Chat Model Error: {e}")

if __name__ == "__main__":
    run_chat_demo()
