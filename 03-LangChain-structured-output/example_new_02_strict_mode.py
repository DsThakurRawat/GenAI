"""
Topic: Structured Output with `strict=True`
What it does: Enforces 100% schema adherence at the API level for providers that support it 
(like OpenAI's Structured Outputs feature).

Why use it:
- Without strict mode, the LLM is just "trying its best" to follow your prompt instructions.
- With `strict=True`, the provider mathematically restricts the token probabilities so the 
  model *cannot* output invalid JSON.
- Note: Pydantic models must be compatible with the provider's strict mode limitations 
  (e.g., no default values, all fields required, no recursive schemas).
"""

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Define the schema. 
# RULE FOR STRICT MODE: All fields must be explicitly required or fully defined. 
# Complex default factories or recursive types might be rejected by the API.
class UserProfile(BaseModel):
    username: str = Field(description="The user's chosen handle")
    age: int = Field(description="The user's age")
    is_active: bool = Field(description="Whether the user is currently active")

model = ChatOpenAI(model="gpt-4o-mini")

# 1. Bind the schema with strict=True
# If the schema is incompatible with OpenAI's strict mode, LangChain will throw 
# a clear error right here during initialization.
strict_model = model.with_structured_output(UserProfile, strict=True)

print("--- Invoking Strict Structured Output ---")
# The LLM is forced to return EXACTLY this schema, nothing else.
user = strict_model.invoke("Create a profile for a 25 year old active user named 'CodeNinja'")

print(type(user))
print(f"Username: {user.username}")
print(f"Age: {user.age}")
print(f"Active: {user.is_active}")
