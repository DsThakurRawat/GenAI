"""
Topic: Model Caching
What it does: Saves the exact prompt and its LLM response. If the identical prompt 
is sent again, the LLM is NOT called; the answer is returned instantly from the cache.

Why use it:
- Saves money (API costs).
- Reduces latency (returns in milliseconds instead of seconds).
- Great during development when you are running the same chain repeatedly.
"""

from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
import time
from dotenv import load_dotenv

load_dotenv()

# Set up an In-Memory cache (stores in RAM, cleared when script exits).
# For production, use `SQLiteCache` or `RedisCache` for persistence!
set_llm_cache(InMemoryCache())

model = ChatOpenAI(model="gpt-4o-mini")
prompt = "Explain quantum computing in one sentence."

print("--- First Call (Cache Miss) ---")
start_time = time.time()
response1 = model.invoke(prompt)
print(f"Time taken: {time.time() - start_time:.2f} seconds")
print(response1.content)

print("\n--- Second Call (Cache Hit) ---")
start_time = time.time()
# This call will hit the InMemoryCache and return instantly!
response2 = model.invoke(prompt)
print(f"Time taken: {time.time() - start_time:.4f} seconds")
print("Notice how much faster it was? And we weren't charged API tokens for the second call!")
