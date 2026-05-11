"""
Topic: Self-Consistency (CoT-SC)
What it does: Runs a Chain of Thought prompt multiple times in parallel and uses 
a majority vote to determine the final correct answer.

Why use it:
- A single Chain of Thought path might accidentally make a math error early on and hallucinate.
- By generating 3 or 5 independent reasoning paths and picking the most common final answer, 
  you drastically improve accuracy on complex reasoning tasks.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv
from collections import Counter
import re

load_dotenv()

# We need some temperature so the model takes slightly different reasoning paths each time!
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = PromptTemplate(
    template="""Think step-by-step to solve this logic puzzle. 
At the very end of your response, output the final numerical answer inside brackets like this: [ANSWER].

Puzzle: {puzzle}""",
    input_variables=["puzzle"]
)

# 1. We create 3 identical chains that run in parallel
# In a real app, you might run 5 or 10 for very hard problems.
reasoning_chain = prompt | model | (lambda msg: msg.content)

parallel_paths = RunnableParallel({
    "path_1": reasoning_chain,
    "path_2": reasoning_chain,
    "path_3": reasoning_chain
})

# 2. A function to extract the [ANSWER] and find the majority vote
def extract_and_vote(results_dict):
    answers = []
    print("\n--- Individual Reasoning Paths ---")
    for key, text in results_dict.items():
        print(f"\n[{key.upper()}]:\n{text}")
        
        # Regex to find [NUMBER]
        match = re.search(r'\[(.*?)\]', text)
        if match:
            answers.append(match.group(1).strip())
    
    if not answers:
        return "Could not extract any answers."
    
    # Count frequencies and get the most common
    most_common = Counter(answers).most_common(1)[0][0]
    return f"\n\n*** MAJORITY VOTE ANSWER: {most_common} ***"

# 3. Build the final chain
self_consistency_chain = parallel_paths | RunnableLambda(extract_and_vote)

print("--- Running Self-Consistency Chain ---")
puzzle_text = "A farmer has 10 sheep. All but 7 die. How many are left?"
result = self_consistency_chain.invoke({"puzzle": puzzle_text})

print(result)
