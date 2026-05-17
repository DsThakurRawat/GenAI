"""
Topic: Few-Shot Chat Prompt Templates
What it does: Implements few-shot learning for modern Chat Models using message lists.

Why use it:
- For `ChatModels`, you shouldn't just concatenate text strings. 
- You should provide examples as a mock conversation history between a 'Human' and an 'AI'.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# 1. Define the examples. Each example represents a Human-AI turn.
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

# 2. Define how each example is formatted into Chat Messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# 3. Create the Few-Shot Chat Template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 4. Assemble the final ChatPromptTemplate using the few-shot template as a piece of it
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt, # <--- The examples get injected here!
        ("human", "{input}"),
    ]
)

print("--- Few-Shot Chat Prompt Result ---")
# .format_messages() returns the actual list of BaseMessage objects to send to the ChatModel.
messages = final_prompt.format_messages(input="What is 5+5?")

for msg in messages:
    print(f"{msg.type.upper()}: {msg.content}")
