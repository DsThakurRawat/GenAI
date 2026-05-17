"""
Topic: Sequential Chains
What it does: Connects multiple simple chains together end-to-end. The output of the first 
chain becomes the input to the second chain.

Explanation:
- You are not limited to just `prompt | model | parser`.
- You can pipe the parsed output of one LLM call directly into the prompt of another.
- This is incredibly powerful for multi-step reasoning or tasks that require splitting up 
  complex prompts into smaller, more focused LLM calls.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

# First step: Generate a detailed report
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# Second step: Summarize the generated report
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text: \n\n {text}',
    input_variables=['text']
)

# Create the sequential chain using LCEL
# The output of the first `parser` (a string) is passed as the 'text' variable to `prompt2`.
chain = prompt1 | model | parser | prompt2 | model | parser

print("--- Invoking Sequential Chain ---")
result = chain.invoke({'topic': 'Unemployment in India'})
print(result)

print("\n--- Chain Graph ---")
chain.get_graph().print_ascii()
