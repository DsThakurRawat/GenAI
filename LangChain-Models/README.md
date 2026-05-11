# LangChain Models Guide

This directory contains examples for interfacing with various Language Models using LangChain.

## 1. LLMs vs Chat Models
- **LLMs (Legacy)**: These models take a single string prompt and return a string completion. Examples: `OpenAI`, `HuggingFaceEndpoint`.
- **Chat Models (Modern)**: These models take a list of `BaseMessage` objects (SystemMessage, HumanMessage, AIMessage) and return an `AIMessage`. Almost all modern models (GPT-4, Claude 3, Gemini 1.5) are optimized for this format. Examples: `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`.

## 2. Embeddings Models
Embeddings models convert text into dense vector representations (lists of floats). This is the foundation of Semantic Search and RAG (Retrieval Augmented Generation).

## 3. Advanced Model Features
To make your applications production-ready, you must go beyond basic inference. I have added new scripts in this directory covering advanced model orchestration:
- **Streaming (`example_new_01_model_streaming.py`)**: Delivering the response chunk-by-chunk to the user interface for a faster perceived load time (`.stream()`).
- **Caching (`example_new_02_model_caching.py`)**: Saving identical API requests to memory or a database to save money and reduce latency.
- **Token Tracking (`example_new_03_token_tracking.py`)**: Intercepting API calls to calculate exact usage to estimate and log costs.
- **Multimodal (`example_new_04_multimodal_inputs.py`)**: Passing images into vision-capable Chat Models (like GPT-4o or Claude 3).
