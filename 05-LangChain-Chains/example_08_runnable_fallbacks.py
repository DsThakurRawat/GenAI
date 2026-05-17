"""
Topic: Runnables Fallbacks (.with_fallbacks)
What it does: Adds fault-tolerance and resilience to your chains. If a primary component fails, 
it automatically tries a backup component.

Explanation:
- API calls can fail due to rate limits, server outages, or model deprecations.
- `with_fallbacks()` lets you define a primary LLM (e.g., an expensive but smart OpenAI model) 
  and a fallback LLM (e.g., an open-source HuggingFace model or a cheaper Anthropic model).
- If the primary model throws an error, the chain gracefully routes the same input to the fallback.
"""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Let's pretend this primary model is broken or rate-limited.
# We will simulate a failure by using a fake, non-existent model name.
primary_model = ChatOpenAI(model="gpt-fake-non-existent-model")

# This is our reliable fallback model.
fallback_model = ChatAnthropic(model_name='claude-3-haiku-20240307')

# Create a prompt and parser
prompt = PromptTemplate(
    template='Give me a brief summary of the book: {book}',
    input_variables=['book']
)
parser = StrOutputParser()

# We attach the fallback directly to the model
resilient_model = primary_model.with_fallbacks([fallback_model])

# Now we build the chain
chain = prompt | resilient_model | parser

print("--- Invoking Resilient Chain with Fallbacks ---")
print("Attempting to call fake OpenAI model... it will fail and fallback to Anthropic.")

try:
    result = chain.invoke({'book': 'The Great Gatsby'})
    print("\nResult received successfully from fallback!")
    print(result)
except Exception as e:
    print(f"Chain failed completely: {e}")
