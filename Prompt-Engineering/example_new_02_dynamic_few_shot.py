"""
Topic: Dynamic Few-Shot Prompting
What it does: Uses a Vector Database (Embeddings) to find the most relevant few-shot 
examples for the specific user input, rather than hardcoding them.

Why use it:
- If you have 100 examples of how to write SQL queries, you can't put all 100 in the prompt 
  (it wastes tokens and confuses the model).
- Instead, you embed them. When the user asks "How do I group by date?", the system 
  finds the 2 examples related to dates and ONLY injects those into the prompt.
"""

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. A large list of potential examples
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

# 2. We use an Example Selector backed by a Vector Store
# It will embed the inputs and search for the closest ones.
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2 # Only pick the TOP 2 most relevant examples!
)

# 3. The template for a single example
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Word: {input}\nAntonym: {output}"
)

# 4. The Few-Shot Template powered by the dynamic selector
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input word.\n",
    suffix="Word: {word}\nAntonym:",
    input_variables=["word"]
)

print("--- Testing Dynamic Selection ---")
# Let's ask for an antonym of a weather-related word.
# It should dynamically select "sunny" and "windy" as the examples!
print(dynamic_prompt.format(word="rainy"))

print("\n--- Testing Another Input ---")
# Let's ask for a feeling.
# It should dynamically select "happy" and "energetic".
print(dynamic_prompt.format(word="depressed"))
