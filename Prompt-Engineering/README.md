# Prompt Engineering Guide

This directory contains a structured curriculum for mastering Prompt Engineering with LangChain.

## Existing Curriculum
- **Level 1 (`level_1_foundations.py`)**: Zero-shot vs Few-shot, Role-playing, Output formatting.
- **Level 2 (`level_2_advanced_reasoning.py`)**: Chain of Thought (CoT), Step-by-step reasoning.
- **Level 3 (`level_3_platform_specific.py`)**: Handling quirks of different models (OpenAI vs Anthropic vs Gemini).
- **Level 4 (`level_4_production_agents.py`)**: ReAct framework, Tool use, and Agentic loops.

## Advanced Techniques Added
I have researched LangChain's advanced prompting techniques and added new scripts:

- **Self-Consistency (`example_new_01_self_consistency.py`)**: An extension of Chain of Thought. Instead of asking the model to reason once, you ask it to reason 3-5 times independently, and then use a majority vote to pick the final answer. This massively reduces hallucinations on math and logic tasks.
- **Dynamic Few-Shot (`example_new_02_dynamic_few_shot.py`)**: Instead of hardcoding 5 examples into your prompt, you store 100 examples in a Vector Database. When a user asks a question, LangChain dynamically searches the database and injects the 3 most relevant examples into the prompt before sending it to the LLM.
