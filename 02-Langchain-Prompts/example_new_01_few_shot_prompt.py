"""
Topic: Few-Shot Prompt Templates
What it does: Constructs a prompt that includes multiple examples of the task before 
asking the final question.

Why use it:
- "Zero-shot" prompting (just asking the question) often fails for complex formats.
- "Few-shot" prompting significantly improves LLM performance and reliability by 
  showing it exactly what a good answer looks like.
"""

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# 1. First, create a template for how a single example should look.
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template="Word: {word}\nAntonym: {antonym}"
)

# 2. Provide the examples themselves as a list of dictionaries.
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
]

# 3. Create the FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input word.\n",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
    example_separator="\n\n"
)

print("--- Few-Shot Prompt Result ---")
# When we format it, LangChain safely stitches the prefix, examples, and suffix together.
formatted_prompt = few_shot_prompt.format(input="massive")
print(formatted_prompt)
