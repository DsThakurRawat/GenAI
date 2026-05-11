"""
Topic: Partial Prompts
What it does: Allows you to "pre-fill" some variables in a PromptTemplate while 
leaving others to be filled in later in the chain.

Why use it:
- Useful when you have data available *now* (like the current date/time) but the 
  rest of the data (like the user's question) won't be available until later.
- Instead of keeping track of the date everywhere, you partially format the prompt early.
"""

from langchain_core.prompts import PromptTemplate
from datetime import datetime

# 1. Let's create a function that always returns the current time
def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")

# 2. We define a prompt with two variables: `date` and `question`.
prompt = PromptTemplate(
    template="Today's date is {date}. Based on this, please answer: {question}",
    input_variables=["question", "date"]
)

print("--- Partial with Function ---")
# We use .partial() to bind the `date` variable to our function.
# Now, the new prompt ONLY requires `question`!
partial_prompt = prompt.partial(date=_get_datetime)

print(partial_prompt.format(question="How many days until Christmas?"))


print("\n--- Partial with Static Value ---")
# You can also partially bind simple strings.
prompt2 = PromptTemplate(
    template="Tell me about {topic} in the style of {person}.",
    input_variables=["topic", "person"]
)

# Pre-fill 'person'
partial_prompt2 = prompt2.partial(person="Shakespeare")

print(partial_prompt2.format(topic="artificial intelligence"))
