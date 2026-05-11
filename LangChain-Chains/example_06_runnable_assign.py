"""
Topic: RunnableAssign (.assign)
What it does: Allows you to add new keys to the dictionary state passing through the chain, 
without removing or modifying the existing keys.

Explanation:
- Often, when you use a RunnableSequence, each step transforms the data, and previous 
  data is lost unless explicitly carried forward.
- `RunnablePassthrough.assign()` solves this by saying: "Keep everything currently in the 
  dictionary, but calculate these *new* values and add them to it."
- This is one of the most critical primitives for building complex data pipelines.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
model = ChatOpenAI()
parser = StrOutputParser()

# Step 1: Define a prompt that generates a short summary
summary_prompt = PromptTemplate(
    template='Summarize this text in 1 sentence: {text}',
    input_variables=['text']
)

# Step 2: Define a prompt that translates text
translation_prompt = PromptTemplate(
    template='Translate the following summary to French: {summary}',
    input_variables=['summary']
)

# Here is the magic:
# 1. We start with {"text": "..."}
# 2. .assign(summary=...) runs the summary chain, and ADDS the 'summary' key to the dictionary.
# 3. Now the dictionary is {"text": "...", "summary": "..."}
# 4. We can pipe this directly into the translation_prompt which expects 'summary'.

chain = (
    {"text": RunnablePassthrough()} 
    | RunnablePassthrough.assign(summary=summary_prompt | model | parser)
    | RunnablePassthrough.assign(french_translation=translation_prompt | model | parser)
)

print("--- Invoking Assign Chain ---")
# When we invoke, the final output will be a dictionary containing ALL three keys!
result = chain.invoke("LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and reason.")

print(f"Original Text: {result['text']}")
print(f"Summary: {result['summary']}")
print(f"French Translation: {result['french_translation']}")
