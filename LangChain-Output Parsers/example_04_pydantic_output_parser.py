"""
Topic: Pydantic Output Parser (PydanticOutputParser)
What it does: Uses the popular `pydantic` library to enforce a very strict schema, 
including nested objects, data types, and validation logic.

Why use it:
This is the most robust parsing method. Pydantic ensures the types are exactly correct 
(e.g., forcing a list of strings, integers). If the LLM makes a mistake, Pydantic throws 
a validation error which can be caught or used with an OutputFixingParser.
"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1. Define the Pydantic Schema
class CountryInfo(BaseModel):
    name: str = Field(description="Name of the country")
    top_cities: list[str] = Field(description="A list of the top 3 largest cities")
    is_island: bool = Field(description="True if the country is an island, False otherwise")

# 2. Initialize the Parser
parser = PydanticOutputParser(pydantic_object=CountryInfo)

# 3. Define the Prompt
template = PromptTemplate(
    template='Provide information about {country}. \n\n{format_instruction}',
    input_variables=['country'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# 4. Create the Chain
chain = template | model | parser

# 5. Invoke the chain
# The result will be a native Pydantic object (CountryInfo).
result = chain.invoke({'country': 'Australia'})

print("--- PydanticOutputParser Result ---")
print(type(result)) # Output: <class '__main__.CountryInfo'>
print(result.name)
print(result.top_cities)
print(result.is_island)
