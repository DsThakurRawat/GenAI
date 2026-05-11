"""
Topic: Structured Output Parser (StructuredOutputParser)
What it does: Allows you to define a specific schema (using ResponseSchema) for the JSON 
output without using the Pydantic library. The parser auto-generates formatting instructions.

Why use it:
It's great when you want to enforce specific keys in a JSON output, along with descriptions 
for what each key should contain. It is highly flexible.
"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1. Define the Schema
# We define exactly what keys we want in our output and what they mean.
response_schemas = [
    ResponseSchema(name='capital', description='The capital city of the country'),
    ResponseSchema(name='population', description='The estimated population of the country'),
    ResponseSchema(name='language', description='The official language(s) spoken')
]

# 2. Initialize the Parser from the schemas
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 3. Define the Prompt
# Inject the strict format instructions generated from our schemas
template = PromptTemplate(
    template='Tell me about {country}. \n\n{format_instruction}',
    input_variables=['country'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# 4. Create the Chain
chain = template | model | parser

# 5. Invoke the chain
# The result will be a dictionary containing exactly the keys defined in the ResponseSchema.
result = chain.invoke({'country': 'Japan'})

print("--- StructuredOutputParser Result ---")
print(type(result)) # Output: <class 'dict'>
print(result)
