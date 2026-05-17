
#how 
from langchain_openai import ChatOpenAI
from src.core.logger import logger
from src.core.config import settings

# Load configuration and initialize the ChatOpenAI model
# Production Tip: Using src.core.config ensures we don't hardcode keys
# We use ChatOpenAI because it's the modern interface for GPT models
"""
OPENAI LLM DEMONSTRATION
========================

This script demonstrates the most basic usage of the OpenAI Chat Model using the 
`.invoke()` method.

### PARAMETER IMPACT GUIDE (Docstring Explanation):

1. **model (str)**:
   - `gpt-3.5-turbo`: Legacy fast/cheap model.
   - `gpt-4o`: Modern standard.
   - **Decision**: Use 3.5 for simple tasks where latency is more important than reasoning.

2. **temperature (float)**:
   - **Impact**: Controls the "sharpness" of the model's word selection. 
   - **Range**: 0.0 to 2.0.
"""

# Load configuration and initialize the ChatOpenAI model
llm = ChatOpenAI(
    # PARAMETER: api_key
    # IMPACT: Authenticates your request to OpenAI.
    api_key=settings.OPENAI_API_KEY,
    
    # PARAMETER: model
    # IMPACT: Selects the specific GPT version.
    model="gpt-3.5-turbo"
)

# how to use llms  
"""
DETAILED EXPLANATION OF .invoke()
---------------------------------
The .invoke() method is the core entry point for any "Runnable" object in LangChain 
(LLMs, Chains, Prompts, etc.). 

How it works internally:
1. Input Parsing: It accepts multiple formats:
   - String: "What is Python?" (Converted to a HumanMessage internally).
   - List of Messages: [SystemMessage(...), HumanMessage(...)] (For chat history).
   - Dict: {"input": "..."} (Common in complex chains).

2. Protocol: It follows the Runnable Interface, which ensures consistency. 
   Whether you are calling a simple LLM or a complex 50-step chain, 
   the method is always .invoke().

3. Output: For ChatModels, it returns an AIMessage object. 
   - .content: The actual text response.
   - .response_metadata: Tokens used, model name, finish reason.
   - .id: Unique identifier for the response.

4. Lifecycle: 
   - Pre-processing -> API Call (OpenAI/Gemini) -> Post-processing -> Output.

Synchronous vs Asynchronous:
- .invoke(): Blocks the thread until the LLM responds.
- .ainvoke(): The async version for high-performance applications (fastapi/asyncio).
"""
# In LangChain, the standard way to interact with an LLM is the .invoke() method.

#invoke method in langchain is important
try:
    logger.info("Asking the LLM: What is the capital of India?")
    
    # We call .invoke() to send our question to the model
    result = llm.invoke("What is the capital of India?")
    
    # The result contains the content, metadata, and more.
    # In production, we log the success and print the answer.
    logger.success("Received response from LLM")
    print(f"\nAnswer: {result.content}")

except Exception as e:
    logger.error(f"Failed to connect to OpenAI: {e}")
    print("\n[TIP] If you don't have an OpenAI key, you can use OpenRouter or Gemini as discussed.")
