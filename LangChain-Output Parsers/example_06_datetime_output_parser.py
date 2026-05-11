"""
Topic: Datetime Output Parser
What it does: Forces the LLM to output a date/time string in a specific format 
(by default: "%Y-%m-%dT%H:%M:%S.%fZ") and parses it into a Python `datetime` object.

Why use it:
LLMs are notoriously bad at outputting consistent date formats. This parser enforces 
a format instruction and automatically converts it so you can perform date math immediately.
"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import DatetimeOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1. Initialize the Parser
parser = DatetimeOutputParser()

# 2. Define the Prompt
template = PromptTemplate(
    template='When was the event "{event}"? \n\n{format_instruction}',
    input_variables=['event'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# 3. Create the Chain
chain = template | model | parser

# 4. Invoke the chain
result = chain.invoke({'event': 'the signing of the Declaration of Independence'})

print("--- DatetimeOutputParser Result ---")
print(type(result)) # Output: <class 'datetime.datetime'>
print(result)
