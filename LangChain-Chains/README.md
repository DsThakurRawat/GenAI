# LangChain Chains and LCEL Guide

This directory covers the core concepts of LangChain: **Chains**, **LCEL** (LangChain Expression Language), and **Runnables**.

## 1. Components of LLM Applications
A typical LLM application consists of several modular components:
- **Models**: The underlying LLM (e.g., OpenAI, Anthropic, HuggingFace).
- **Prompts**: Templates that format user input into instructions for the LLM.
- **Output Parsers**: Tools that extract and format the LLM's raw text response into structured data (JSON, Pydantic objects, etc.).
- **Chains**: The glue that connects these components together into a predictable workflow.

## 2. What are LangChain Chains?
Historically, a "Chain" in LangChain was a Python class (like `LLMChain` or `SequentialChain`) that executed a series of calls to LLMs, tools, or data preprocessing steps. 
In modern LangChain, Chains are predominantly built using **LCEL** (LangChain Expression Language).

## 3. LCEL (LangChain Expression Language)
LCEL is a declarative way to easily compose chains together. It uses the pipe operator (`|`) to pass the output of one component directly as the input to the next.
Example: `chain = prompt | model | parser`

## 4. Runnables and Runnable Primitives
The core abstraction in modern LangChain is the `Runnable` interface. Any component you can chain together using LCEL (Prompts, Models, Parsers) implements the `Runnable` protocol. This gives them standard methods like `.invoke()`, `.stream()`, and `.batch()`.

### Common Runnable Primitives:
- **`RunnableSequence`**: Created automatically when you use the `|` operator. Runs components in order.
- **`RunnableParallel` / `RunnableMap`**: Runs multiple Runnables at the same time and combines their outputs into a dictionary.
- **`RunnableBranch`**: Provides `if/else` conditional logic within a chain.
- **`RunnableLambda`**: Converts a custom Python function into a Runnable so it can be used in a chain.
- **`RunnablePassthrough`**: Passes inputs completely unchanged or adds new keys to the input dictionary.

### Advanced LCEL Primitives:
- **`.assign()`**: Appends new key-value pairs to the dictionary output of a previous step without losing the original data. Extremely useful for maintaining context.
- **`.bind()`**: Binds specific parameters (like stop sequences, tools, or model kwargs) to a Runnable at runtime.
- **`.with_fallbacks()`**: Adds resilience to your application. If a primary model/runnable fails (e.g. rate limit), it automatically switches to a backup runnable.

## 5. How Runnables and Chains Work Together
When you build an application, you are essentially snapping together different `Runnable` puzzle pieces using LCEL. The chain orchestrates the flow of data. 

For deeper explanations and executable code examples, refer to the numbered `example_xx_...py` Python scripts provided in this directory.
