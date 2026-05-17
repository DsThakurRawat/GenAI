"""
UNIVERSAL LLM PARAMETER GUIDE
=============================

This guide explains all parameters used in LangChain Chat Models across different providers 
(OpenAI, Anthropic, Google Gemini, and Groq).

### 1. MODEL IDENTITY (model)
- **What it is**: The ID of the specific model to use.
- **Variations**:
    - **OpenAI**: `gpt-4o`, `gpt-3.5-turbo`
    - **Anthropic**: `claude-3-5-sonnet-20240620`, `claude-3-opus-20240229`
    - **Google**: `gemini-1.5-flash`, `gemini-1.5-pro`
    - **Groq**: `llama3-70b-8192`, `mixtral-8x7b-32768`
- **Response Impact**: This is the single biggest factor in the quality of the response. 
  Larger models (Pro/Opus/70b) handle complex logic better; smaller models (Flash/Sonnet/8b) 
  are faster and cheaper.

### 2. CREATIVITY & SAMPLING (temperature, top_p)
- **Temperature (0.0 to 2.0)**:
    - **0.0 (Greedy)**: The model always picks the absolute most likely next token.
    - **1.0+ (Wild)**: The model starts picking less likely tokens.
    - **Impact**: Use 0.0 for code, math, and structured data extraction. Use 0.7-1.0 for creative writing.
- **Top P (0.0 to 1.0)**:
    - **What it is**: Nucleus sampling. It limits choices to a "pool" of top words.
    - **Impact**: Lowering this (e.g., to 0.1) makes the model more "focused" and less likely to wander.

### 3. RESPONSE LIMITS (max_tokens)
- **What it is**: The maximum length of the AI's response.
- **Impact**: If set too low, the response will stop mid-sentence. If set too high, 
  you risk high costs or the model "hallucinating" filler text.

### 4. PENALTIES (frequency_penalty, presence_penalty)
- **Frequency Penalty (-2.0 to 2.0)**:
    - **Impact**: Positive values penalize new tokens based on their existing frequency in the text. 
      Prevents the model from repeating the same words/phrases.
- **Presence Penalty (-2.0 to 2.0)**:
    - **Impact**: Positive values penalize new tokens based on whether they have appeared so far. 
      Encourages the model to talk about NEW topics.

### 5. CONTROL SEQUENCES (stop)
- **What it is**: A list of strings that tell the model to stop generating.
- **Example**: `stop=["###", "Question:"]`.
- **Impact**: Useful for few-shot prompting where you want the model to generate one example and then stop.

### 6. PROVIDER SPECIFIC PARAMETERS
- **Google Gemini**: `safety_settings` (to avoid blocking content).
- **Anthropic**: `max_tokens` is mandatory in many calls.
- **Groq**: `max_retries` (important for their aggressive rate limits).
"""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from src.core.config import settings

def explain_variation_example():
    """
    Example of how changing parameters changes the response.
    """
    # DETERMINISTIC MODEL (For JSON extraction)
    extractor = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.0,  # CRITICAL: No creativity allowed
        max_tokens=100
    )

    # CREATIVE MODEL (For Poem generation)
    poet = ChatGroq(
        model="llama3-70b-8192",
        temperature=1.2,  # HIGH: Encourages unique word choices
        max_tokens=500
    )

    print("Variation Guide:")
    print("- Need a JSON? Set temperature=0.0")
    print("- Need a Story? Set temperature=0.8")
    print("- Need to save money? Use '8b' or 'flash' models.")

if __name__ == "__main__":
    explain_variation_example()
