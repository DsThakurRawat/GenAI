# LangChain Prompts Guide

This directory covers LangChain's Prompt Engineering tools and templates.

## 1. Core Concepts
- **`PromptTemplate`**: Used for standard string-in, string-out LLMs.
- **`ChatPromptTemplate`**: Used for modern Chat Models. It accepts a list of `SystemMessage`, `HumanMessage`, and `AIMessage` objects (or their string template equivalents).
- **`MessagesPlaceholder`**: A dynamic insertion point in a `ChatPromptTemplate` where you can inject a variable-length list of messages (essential for memory/chat history).

## 2. Advanced Prompting Techniques
I have researched the LangChain documentation and added scripts to demonstrate the following advanced techniques:

- **Few-Shot Prompting (`example_new_01_few_shot_prompt.py`)**: Giving the model a few examples of inputs and desired outputs *before* asking the real question.
- **Few-Shot Chat Prompting (`example_new_02_few_shot_chat_prompt.py`)**: How to implement few-shot examples safely using ChatMessages (System/Human/AI).
- **Partial Prompts (`example_new_03_partial_prompts.py`)**: Binding some variables early in the code (e.g. current date/time) before passing the prompt further down the chain.
- **Pipeline Prompts (`example_new_04_pipeline_prompts.py`)**: Composing a master prompt out of multiple smaller prompt templates (e.g., combining an `introduction`, an `example`, and the `main_task` templates together).
