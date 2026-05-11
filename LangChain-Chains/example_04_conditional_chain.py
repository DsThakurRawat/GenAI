"""
Topic: Conditional Chains (RunnableBranch)
What it does: Routes the data to different sub-chains based on conditional logic (If/Else).

Explanation:
- `RunnableBranch` evaluates a list of (condition, runnable) pairs.
- It runs the `condition` (which should return True or False). If True, it executes that `runnable`.
- It acts like an `if-elif-else` block for your chains.
- Often used with a classifier LLM call at the beginning to decide how to route the request.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
model = ChatOpenAI()
parser = StrOutputParser()

# 1. Define schema for the classifier output
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# 2. Classifier Chain
prompt1 = PromptTemplate(
    template='Classify sentiment of this feedback into positive or negative: \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)
classifier_chain = prompt1 | model | parser2

# 3. Define the sub-chains for routing
prompt2 = PromptTemplate(template='Write a thank you note for this positive feedback: {feedback}', input_variables=['feedback'])
prompt3 = PromptTemplate(template='Write an apology for this negative feedback: {feedback}', input_variables=['feedback'])

# 4. Data Routing using RunnablePassthrough
# A common issue with branches is keeping track of the original input. 
# We build a dict that contains BOTH the original 'feedback' string and the classified 'sentiment'.
routing_setup = {
    "feedback": RunnablePassthrough(), # Keeps the original string
    "sentiment": classifier_chain      # Runs the classifier
}

# 5. The Branching Logic
# Now 'x' is the dict we built in step 4!
branch_chain = RunnableBranch(
    (lambda x: x["sentiment"].sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x["sentiment"].sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not classify sentiment") # Default fallback
)

# 6. Combine into a master chain
chain = routing_setup | branch_chain

print("--- Invoking Conditional Chain ---")
result = chain.invoke({'feedback': 'This is a beautiful phone'})
print(result)
