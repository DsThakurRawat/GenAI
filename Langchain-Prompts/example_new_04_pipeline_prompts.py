"""
Topic: Pipeline Prompt Templates
What it does: Composes multiple smaller `PromptTemplates` together into one master template.

Why use it:
- Keeps your code DRY (Don't Repeat Yourself).
- If you have standard components (like a standard introduction, a standard formatting block, etc.), 
  you don't want to copy-paste them into every prompt. You can build a "Pipeline" of prompts instead.
"""

from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate

# 1. The Full Master Template
# It has "placeholders" for the sub-prompts.
full_template = PromptTemplate.from_template(
    """{introduction}
    
{example}

{start}"""
)

# 2. The Sub-Templates
introduction_template = PromptTemplate.from_template(
    "You are impersonating {person}."
)
example_template = PromptTemplate.from_template(
    "Here's an example of an interaction:\nQ: {example_q}\nA: {example_a}"
)
start_template = PromptTemplate.from_template(
    "Now, do this for real!\nQ: {input}\nA:"
)

# 3. Assemble the Pipeline
# We map the names in the full_template to the actual sub-templates
input_prompts = [
    ("introduction", introduction_template),
    ("example", example_template),
    ("start", start_template),
]

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_template, 
    pipeline_prompts=input_prompts
)

print("--- Pipeline Prompt Result ---")
# Look at how many variables are required now! 
# It merges all the variables from all sub-templates into one.
print("Required variables:", pipeline_prompt.input_variables)

result = pipeline_prompt.format(
    person="Yoda",
    example_q="What is your name?",
    example_a="Yoda, my name is.",
    input="How do I learn Python?"
)

print("\n" + result)
