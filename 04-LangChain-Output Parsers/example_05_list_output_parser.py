"""
Topic: Comma Separated List Output Parser
What it does: It asks the LLM to output a simple comma-separated string 
(e.g., "apple, banana, orange") and then automatically parses it into a Python list.

Why use it:
Great for when you need a simple list of items (like tags, keywords, or names) 
and don't want the overhead of JSON parsing.
"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1. Initialize the Parser
parser = CommaSeparatedListOutputParser()

# 2. Define the Prompt
template = PromptTemplate(
    template='List 5 common {category}. \n\n{format_instruction}',
    input_variables=['category'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# 3. Create the Chain
chain = template | model | parser

# 4. Invoke the chain
result = chain.invoke({'category': 'programming languages'})

print("--- CommaSeparatedListOutputParser Result ---")
print(type(result)) # Output: <class 'list'>
print(result)
