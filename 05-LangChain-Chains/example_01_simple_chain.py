"""
Topic: Simple Chain & LCEL Basics
What it does: Demonstrates the most fundamental LangChain Expression Language (LCEL) chain.

Explanation:
- LCEL uses the pipe operator `|` to pass data from one component to another.
- All components in LCEL implement the `Runnable` protocol, giving them common 
  methods like `.invoke()`, `.batch()`, and `.stream()`.
- The chain works like this: Dictionary Input -> Prompt -> Model -> Output Parser -> Final Output
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. The Prompt (Runnable)
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# 2. The Model (Runnable)
model = ChatOpenAI()

# 3. The Output Parser (Runnable)
parser = StrOutputParser()

# 4. The Chain (RunnableSequence)
# This creates a single cohesive pipeline.
chain = prompt | model | parser

# 5. Execution
# We use `.invoke()` to trigger the chain with our starting inputs.
print("--- Invoking Simple Chain ---")
result = chain.invoke({'topic': 'cricket'})
print(result)

# Bonus: You can inspect the computation graph!
print("\n--- Chain Graph ---")
chain.get_graph().print_ascii()
