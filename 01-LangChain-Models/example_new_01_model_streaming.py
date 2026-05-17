"""
Topic: Model Streaming
What it does: Streams the model's response back token-by-token instead of waiting 
for the entire output to be generated.

Why use it:
- Dramatically improves user experience (UX) in chat interfaces. 
- The user sees text immediately (Time to First Token - TTFT) rather than waiting 
  5-10 seconds for a long completion.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# We don't necessarily need to pass `streaming=True` into the constructor anymore in modern LangChain.
# We just use the `.stream()` method.
model = ChatOpenAI(model="gpt-4o-mini")

prompt_text = "Write a 3 paragraph story about a robot discovering a garden."

print("--- Streaming Response ---")
# `.stream()` returns a generator that yields AIMessageChunks
for chunk in model.stream(prompt_text):
    # We print the chunk content directly to the console as it arrives.
    # flush=True ensures the terminal updates immediately without buffering.
    print(chunk.content, end="", flush=True)

print("\n\n--- Finished Streaming ---")
