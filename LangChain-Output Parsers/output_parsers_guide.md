 # LangChain Output Parsers: Comprehensive Guide

Output parsers are classes that help structure language model responses. While LLMs output text (strings), our applications often need structured data (like JSON, lists, dictionaries, or object models). Output parsers bridge this gap by providing instructions to the LLM on how to format its output and then parsing that output into the desired data type.

---

## 1. String Output Parser (`StrOutputParser`)
The `StrOutputParser` is the simplest and most commonly used parser. 
- **What it does:** It takes the output from an LLM (which is usually an `AIMessage` object in modern LangChain) and extracts just the raw string content.
- **Use Case:** When you just want the plain text response from the LLM (e.g., standard chatbots, text generation).
- **Import:** `from langchain_core.output_parsers import StrOutputParser`

---

## 2. JSON Output Parser (`JsonOutputParser`)
The `JsonOutputParser` forces the LLM to output valid JSON and converts it into a Python dictionary.
- **What it does:** It parses a JSON string returned by the LLM into a structured Python dictionary.
- **How to use it:** You must explicitly tell the LLM to output JSON in the prompt. You can optionally provide a Pydantic schema to the `JsonOutputParser` to define exactly what keys the JSON should have.
- **Use Case:** When you need key-value pair data extraction but prefer working with native Python dictionaries.
- **Import:** `from langchain_core.output_parsers import JsonOutputParser`

---

## 3. How to Enforce Schema: Structured Output Parsers
If you want the LLM to return data in a highly specific format, you need to enforce a schema. The `StructuredOutputParser` is one way to do this without using Pydantic.
- **What it does:** You define multiple `ResponseSchema` objects (each with a name, description, and type). The parser generates detailed formatting instructions based on these schemas.
- **How it works:** You inject `parser.get_format_instructions()` into your prompt. The LLM reads these instructions and outputs a markdown JSON block, which the parser then converts into a dictionary.
- **Use Case:** When you want strict JSON schemas but cannot or prefer not to use Pydantic models.
- **Import:** `from langchain.output_parsers import StructuredOutputParser, ResponseSchema`

---

## 4. Pydantic Output Parsers (`PydanticOutputParser`)
The `PydanticOutputParser` is the most robust and widely used method for enforcing complex schemas.
- **What it does:** You define your schema using a Pydantic `BaseModel`. The parser generates prompt instructions based on the model's fields, types, and descriptions, and parses the result directly into an instance of your Pydantic class.
- **Advantages:** Native type validation, deeply nested object support, and autocomplete in modern IDEs. If the LLM misses a required field, Pydantic will throw a clear validation error.
- **Import:** `from langchain_core.output_parsers import PydanticOutputParser`

---

## 5. Difference Between LangChain Core vs. LangChain Pydantic Output Parsers
- **LangChain Core Parsers (`BaseOutputParser`, `StrOutputParser`, `JsonOutputParser`):** 
  These are foundational interfaces and basic parsers built directly into `langchain-core`. They handle standard parsing logic (extracting text, parsing native JSON strings) and do not intrinsically rely on heavy external validation libraries to define the data structures.
- **Pydantic Output Parsers:** 
  While technically located in `langchain-core` now, they strictly depend on the `pydantic` library. They are deeply integrated with Pydantic's data validation engine. The core difference is **Validation**: Core parsers just extract or format data, whereas Pydantic parsers actively validate the data types and enforce strict type constraints before returning the object to you.

---

## 6. Other Types of Output Parsers
LangChain provides several other niche parsers for specific data formats:
1. **`CommaSeparatedListOutputParser`:** Instructs the LLM to return a comma-separated string and parses it into a Python `list`.
2. **`DatetimeOutputParser`:** Forces the LLM to output a date/time string in a specific format and parses it into a Python `datetime` object.
3. **`XMLOutputParser`:** Asks the LLM to output XML and parses it into a nested dictionary format. Useful for older systems or specific prompt techniques (like Anthropic's Claude which excels at XML).
4. **`EnumOutputParser`:** Enforces that the LLM's output string strictly matches one of the values provided in a Python `Enum`.
5. **`OutputFixingParser`:** A special "wrapper" parser. If a primary parser (like Pydantic) fails because the LLM made a syntax error, the `OutputFixingParser` takes the broken string, feeds it back to the LLM along with the error message, and asks the LLM to fix it.
