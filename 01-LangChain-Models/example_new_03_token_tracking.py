"""
Topic: Token Usage Tracking
What it does: Intercepts the API call to standard providers (like OpenAI) and 
records exactly how many prompt tokens, completion tokens, and total cost were incurred.

Why use it:
- Essential for production apps to monitor billing.
- Allows you to implement rate limiting per-user in your own databases based on cost.
"""

from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
prompt = "Translate this into French: Hello, I am a software engineer."

print("--- Tracking Tokens ---")

# We use a context manager to wrap the LLM call.
# Any OpenAI model calls made inside this block will be tracked.
with get_openai_callback() as cb:
    response = model.invoke(prompt)
    
    print("\n--- Model Response ---")
    print(response.content)
    
    print("\n--- Usage Data ---")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost:.6f}")
    
# NOTE: Different providers have different callbacks. Anthropic usage can be retrieved
# directly from the `response.usage_metadata` attribute in modern LangChain.
print("\n--- Native Usage Metadata (Anthropic & OpenAI) ---")
print(response.usage_metadata)
