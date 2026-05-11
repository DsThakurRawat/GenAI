"""
Topic: Structured Output with `include_raw=True`
What it does: Instead of just returning the parsed Pydantic object, it returns a 
dictionary containing the parsed object, the raw AIMessage, and any parsing errors.

Why use it:
- If `include_raw=False` (the default), a parsing error throws an exception and crashes your app.
- If `include_raw=True`, it catches the exception and returns it to you in the dictionary.
- This allows you to inspect the `raw` message to see *why* the LLM failed, log the error, 
  or pass the raw message to an `OutputFixingParser` or retry loop.
"""

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Define the desired schema
class Recipe(BaseModel):
    name: str = Field(description="Name of the dish")
    ingredients: list[str] = Field(description="List of ingredients")
    cook_time_minutes: int = Field(description="Time to cook in minutes")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. Bind the schema with include_raw=True
structured_model = model.with_structured_output(Recipe, include_raw=True)

print("--- Invoking Structured Output (include_raw=True) ---")
# Invoke the model
response = structured_model.invoke("How do you make a quick omelet?")

# 2. Inspect the returned dictionary
print("\nThe response is now a dictionary, not just a Pydantic object!")
print(f"Keys: {response.keys()}")

# 3. Handle potential errors gracefully
if response["parsing_error"] is not None:
    print(f"\n[ERROR] Parsing failed: {response['parsing_error']}")
    print(f"Raw output was: {response['raw'].content}")
else:
    print("\n[SUCCESS] Parsing succeeded!")
    recipe = response["parsed"]
    print(f"Recipe Name: {recipe.name}")
    print(f"Cook Time: {recipe.cook_time_minutes} mins")
    print(f"Ingredients: {recipe.ingredients}")

    # You still have access to the raw token usage, etc!
    print(f"\nRaw Token Usage: {response['raw'].usage_metadata}")
