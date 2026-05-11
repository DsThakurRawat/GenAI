"""
Topic: RunnableBind (.bind)
What it does: Allows you to attach specific keyword arguments (kwargs) to a Runnable 
that will be passed to it when invoked.

Explanation:
- Sometimes you need to pass specific parameters to a model, like a `stop` sequence to tell it 
  when to stop generating text, or maybe configuring `temperature` dynamically.
- You can't put these arguments in the prompt. They belong to the model.
- `.bind()` lets you attach these arguments to the model inside the chain.
- This is heavily used in Tool Calling (Agents), via `model.bind_tools()`.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# We start with a standard model
base_model = ChatOpenAI(temperature=0.7)

# We BIND a specific stop sequence to it. 
# Whenever this bound model is called, it will stop generating if it outputs the word "However".
bound_model = base_model.bind(stop=["However"])

prompt = PromptTemplate(
    template='Write a paragraph about the benefits and drawbacks of Artificial Intelligence. Topic: {topic}',
    input_variables=['topic']
)

chain = prompt | bound_model | StrOutputParser()

print("--- Invoking Bound Chain ---")
# The LLM will likely try to write about drawbacks using the word "However", but the 
# bound stop sequence will instantly halt generation!
result = chain.invoke({'topic': 'AI'})
print(result)
