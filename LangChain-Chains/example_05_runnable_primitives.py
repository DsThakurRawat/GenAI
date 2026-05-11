"""
Topic: Runnable Primitives (RunnableLambda & RunnablePassthrough)
What it does: Utilities to manipulate data flow within an LCEL pipeline without needing an LLM.

Explanation:
- `RunnableLambda`: Wraps any standard Python function into a Runnable. 
  This allows you to insert custom data transformations inside your `|` chain.
- `RunnablePassthrough`: Passes the input unchanged. Very useful when combining inputs 
  in a RunnableParallel or assigning new keys to an existing dictionary input.
"""

from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# --- 1. RunnableLambda Example ---
def uppercase_string(text: str) -> str:
    return text.upper()

# Wrap the function
uppercase_runnable = RunnableLambda(uppercase_string)

print("--- RunnableLambda ---")
print(uppercase_runnable.invoke("hello world")) 
# Output: HELLO WORLD


# --- 2. RunnablePassthrough Example ---
# Imagine we have an input string, but our next prompt expects a dictionary with a 'text' key.
# We can use RunnableParallel + RunnablePassthrough to build it on the fly!

builder_chain = RunnableParallel({
    "original_text": RunnablePassthrough(), # Keeps the exact input
    "uppercased_text": uppercase_runnable   # Transforms the input
})

print("\n--- RunnablePassthrough with Parallel ---")
# If we invoke with a string...
result = builder_chain.invoke("my secret message")
print(result)
# Output: {'original_text': 'my secret message', 'uppercased_text': 'MY SECRET MESSAGE'}
