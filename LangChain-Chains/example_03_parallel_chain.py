"""
Topic: Parallel Chains (RunnableParallel)
What it does: Executes multiple runnables simultaneously and combines their outputs.

Explanation:
- `RunnableParallel` allows you to split the data flow.
- If you have independent tasks (e.g., generating notes AND generating a quiz from the same text), 
  you don't have to wait for one to finish before starting the other.
- It returns a dictionary where keys are the names you defined, and values are the results of 
  the corresponding runnables.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

# Parallel Task 1: Generate Notes
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

# Parallel Task 2: Generate Quiz
prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

# Task 3: Merge them
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

# This executes both prompt1|model|parser and prompt2|model|parser concurrently!
parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model | parser
})

# The output of parallel_chain is a dict: {'notes': '...', 'quiz': '...'}
# This dict matches perfectly with the input variables expected by prompt3.
chain = parallel_chain | prompt3 | model | parser

text = "Support vector machines (SVMs) are a set of supervised learning methods used for classification..."

print("--- Invoking Parallel Chain ---")
result = chain.invoke({'text': text})
print(result)
