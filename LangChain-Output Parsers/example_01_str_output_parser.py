"""
Topic: String Output Parser (StrOutputParser)
What it does: This is the most basic output parser. Language models typically return a message 
object (like AIMessage). The StrOutputParser simply extracts the raw text (.content) from this 
message object and returns it as a string.

Why use it:
It's the easiest way to interface with an LLM when you just want plain text output. 
It belongs to langchain_core.
"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 1. Initialize the Parser
# It doesn't require any schemas or format instructions because it just extracts the text.
parser = StrOutputParser()

# 2. Define the Prompt
template = PromptTemplate(
    template='Tell me a short joke about {topic}',
    input_variables=['topic'],
)

# 3. Create the Chain
chain = template | model | parser

# 4. Invoke the chain
# The result will be a raw Python string, extracted from the model's AIMessage.
result = chain.invoke({'topic': 'programming'})

print("--- StrOutputParser Result ---")
print(type(result)) # Output: <class 'str'>
print(result)
