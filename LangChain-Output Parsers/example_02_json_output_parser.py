"""
Topic: JSON Output Parser (JsonOutputParser)
What it does: It forces the LLM to return its response formatted as a JSON string, 
and then parses that string into a native Python dictionary.

Why use it:
Useful when you want data in a key-value format without strictly defining the types 
(like Pydantic). It relies on the LLM's ability to output valid JSON.

Note: You must explicitly tell the model to output JSON in your prompt!
"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1. Initialize the Parser
parser = JsonOutputParser()

# 2. Define the Prompt
# It is CRITICAL to inject the format_instructions into the prompt so the LLM 
# knows it has to output JSON.
template = PromptTemplate(
    template='Give me 3 facts about {topic}. \n\n{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# 3. Create the Chain
chain = template | model | parser

# 4. Invoke the chain
# The result will be parsed into a Python dictionary.
result = chain.invoke({'topic': 'black holes'})

print("--- JsonOutputParser Result ---")
print(type(result)) # Output: <class 'dict'>
print(result)
