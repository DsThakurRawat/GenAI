from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.core.logger import logger
from src.core.config import settings

"""
GROQ CHAT MODEL DEMONSTRATION
============================

This script demonstrates how to initialize and use the Groq Chat Model in LangChain.
It also serves as a guide for common LLM parameters.

### CORE PARAMETERS GUIDE (Docstring Explanation):

1. **model (str)**:
   - *What it is*: The specific LLM "brain" you are calling.
   - *Variations*: 
     - "llama3-70b-8192": Large, slow, extremely smart.
     - "llama3-8b-8192": Small, ultra-fast, good for classification.
   - *Decision*: Choose 70b for complex reasoning and 8b for simple speed.

2. **temperature (float)**:
   - *What it is*: Controls the degree of "unpredictability" or creativity.
   - *Range*: 0.0 to 2.0 (usually kept between 0.0 and 1.0).
   - *Variations*:
     - 0.0: Deterministic. Always picks the most likely word. Best for coding/math.
     - 0.7: Balanced. Good for general chat.
     - 1.0+: Creative. Best for poems, stories, or unique ideas.
   - *Decision*: Low temp for accuracy; High temp for creativity.

3. **max_tokens (int)**:
   - *What it is*: Hard limit on response length.
   - *Variations*: 
     - Low (e.g., 50): Forces concise answers.
     - High (e.g., 4000): Allows long essays.
   - *Decision*: Use to control costs and prevent the model from rambling.

4. **top_p (float)**:
   - *What it is*: Nucleus sampling. Limits the model to a "pool" of top words whose cumulative probability = P.
   - *Decision*: Set to 1.0 to consider all words; set lower (e.g., 0.1) to be very selective.

5. **stop (list)**:
   - *What it is*: Sequences that tell the model to "shut up" immediately.
   - *Example*: ["\n", "User:"] - stops generating if it sees a newline or "User:".
"""


# Initialize Groq Chat Model
# Groq is compatible with the ChatOpenAI interface, but LangChain provides a dedicated ChatGroq class.
chat_model = ChatGroq(
    groq_api_key=settings.GROQ_API_KEY,
    
    # PARAMETER: model
    # EXPLANATION: Specifies which LLM to use.
    # IMPACT: 
    # - llama3-70b-8192: High reasoning capabilities, slower than 8b, but much more accurate.
    # - llama3-8b-8192: Extremely fast, good for simple tasks, lower cost/latency.
    model="llama3-70b-8192", 
    
    # PARAMETER: temperature (0.0 to 2.0)
    # EXPLANATION: Controls the randomness/creativity of the response.
    # IMPACT:
    # - 0.0 (Deterministic): Use for factual tasks, coding, or data extraction.
    # - 0.7 - 1.0 (Creative): Use for brainstorming or storytelling.
    # EXAMPLE: At 0.0, "The capital of France is..." will ALWAYS be "Paris".
    temperature=0.5,

    # PARAMETER: max_tokens
    # EXPLANATION: The maximum number of tokens to generate.
    # IMPACT: Prevents "runaway" responses and controls cost.
    max_tokens=None, 

    # PARAMETER: top_p (Nucleus Sampling)
    # EXPLANATION: Alternative to temperature. Considers the top P% of probability mass.
    # IMPACT: 1.0 means all words, 0.1 means only top 10% most likely.
    top_p=1.0,

    # PARAMETER: max_retries
    # EXPLANATION: Retries if the API call fails.
    max_retries=2,
)

def run_groq_demo():
    try:
        logger.info("Initializing Groq Demo...")
        
        messages = [
            SystemMessage(content="You are a helpful assistant that provides extremely concise answers."),
            HumanMessage(content="Explain quantum entanglement in 2 sentences.")
        ]
        
        logger.info("Sending request to Groq (Llama 3)...")
        response = chat_model.invoke(messages)
        
        logger.success("Response received from Groq.")
        print("\n--- Groq Response (Llama 3 70B) ---")
        print(response.content)
        print("-----------------------------------\n")

    except Exception as e:
        logger.error(f"Groq API Error: {e}")
        print("\n[TIP] Make sure GROQ_API_KEY is set in your .env file.")
        print("[TIP] You can get a key at https://console.groq.com/keys")

if __name__ == "__main__":
    run_groq_demo()
