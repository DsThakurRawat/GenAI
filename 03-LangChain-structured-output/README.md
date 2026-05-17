# LangChain Structured Output Guide

This directory covers `.with_structured_output()`, LangChain's standard interface for forcing Chat Models to return data matching a specific schema (Pydantic, TypedDict, or JSON Schema).

## 1. Core Implementations
You already have excellent examples of the core functionality:
- `with_structured_output_pydantic.py` (Using Pydantic classes)
- `with_structured_output_typeddict.py` (Using Python TypedDicts)
- `with_structured_output_json.py` (Using raw JSON schema dictionaries)
- `with_structured_output_llama.py` (Provider specific example)

## 2. Advanced Parameters
I have researched the LangChain documentation and added new scripts demonstrating how to make structured outputs robust for production:

- **Handling Exceptions and Raw Output (`example_new_01_include_raw.py`)**: 
  By default, if the LLM fails to match your schema, LangChain throws an error and the entire response is lost. Passing `include_raw=True` forces LangChain to return a dictionary containing the parsed output, the raw message, and any parsing errors so you can handle them gracefully.
  
- **Strict Mode (`example_new_02_strict_mode.py`)**: 
  Some providers (like OpenAI) support a "Strict JSON Mode" which guarantees 100% adherence to your schema by mathematically restricting the tokens the model can output. You enable this by passing `strict=True`.
