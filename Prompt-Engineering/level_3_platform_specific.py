"""
Level 3: Architecture & System-Specific Best Practices

While foundational prompt engineering works across all LLMs, every major AI lab 
fine-tunes their models differently. To get the absolute best performance, you must 
tailor your syntax to match the model's training data.

This module covers the officially recommended prompt design strategies from:
1. OpenAI (GPT-4)
2. Anthropic (Claude 3)
3. Google (Gemini / Vertex AI)
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='\n%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# We use Gemini for execution, but the string formatting demonstrates the labs' recommendations.
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)


def demonstrate_openai_strategies():
    """
    1. OpenAI Strategies (GPT models)
    
    Key Recommendations from OpenAI:
    - Delimiters: Use ```, \"\"\", < >, or XML tags to clearly separate instructions from data.
    - Explicit Fallbacks: Tell the model exactly what to do if it CANNOT fulfill the request 
      to prevent hallucinations.
    - System Messages: GPT models pay heavy attention to the System Message. Use it to set 
      the core persona and unbreakable rules.
    """
    logger.info("="*50)
    logger.info("1. OPENAI STRATEGY: DELIMITERS & FALLBACKS")
    logger.info("="*50)
    
    # Notice the explicit fallback instruction and the triple backticks
    openai_prompt = PromptTemplate.from_template("""
    Your task is to extract the date of the meeting from the text.
    
    [RULE] If the text does not contain a meeting date, you MUST output exactly: "NO_DATE_FOUND"
    
    Text:
    ```
    {user_text}
    ```
    """)
    
    text_with_date = "Hey team, just confirming our sync on October 14th at 2PM."
    text_no_date = "Hey team, just checking in on the project status."
    
    res1 = (openai_prompt | model).invoke({"user_text": text_with_date})
    logger.info(f"Input with date -> Output: {res1.content.strip()}")
    
    res2 = (openai_prompt | model).invoke({"user_text": text_no_date})
    logger.info(f"Input without date -> Output: {res2.content.strip()}")


def demonstrate_anthropic_strategies():
    """
    2. Anthropic Strategies (Claude models)
    
    Key Recommendations from Anthropic:
    - XML Tagging: Claude is explicitly fine-tuned to obey XML tags. You should structure 
      your entire prompt using them (e.g., <document>, <instructions>, <example>).
    - The <scratchpad>: Claude performs much better if you tell it to "think" inside 
      <scratchpad> tags before outputting the final answer. This is Anthropic's version of CoT.
    - Prefilling: (Note: Harder to show in standard LangChain, but you pass the start of the 
      assistant's response to force formatting).
    """
    logger.info("\n" + "="*50)
    logger.info("2. ANTHROPIC STRATEGY: XML TAGGING & SCRATCHPAD")
    logger.info("="*50)
    
    anthropic_prompt = PromptTemplate.from_template("""
    <instructions>
    You are an AI assistant. Analyze the document provided. 
    First, use the <scratchpad> tags to outline your thoughts.
    Second, output a 1-sentence summary inside <summary> tags.
    </instructions>
    
    <document>
    {document}
    </document>
    """)
    
    doc = "Quantum computing relies on qubits, which can exist in multiple states simultaneously due to superposition. This allows quantum computers to process massive amounts of possibilities at once, unlike classical bits which are strictly 0 or 1."
    
    logger.info("Prompting with XML structure...")
    result = (anthropic_prompt | model).invoke({"document": doc})
    logger.info(result.content.strip())


def demonstrate_google_strategies():
    """
    3. Google Strategies (Gemini / Vertex AI)
    
    Key Recommendations from Google:
    - Prompt Chaining: Instead of writing a massive "megaprompt", break complex tasks 
      into sequential chains. Output of Prompt A becomes Input of Prompt B.
    - Parameter Tuning: Actively adjust Temperature, Top-K, and Top-P based on the task.
      (e.g., 0.0 for extraction, 0.8 for creative ideation).
    """
    logger.info("\n" + "="*50)
    logger.info("3. GOOGLE STRATEGY: PROMPT CHAINING")
    logger.info("="*50)
    
    logger.info("Task: Read a text, extract key entities, then write a poem about them.")
    logger.info("Instead of one prompt, we chain two prompts together.")
    
    text = "The quick brown fox jumped over the lazy dog."
    
    # Prompt A: Extraction
    extract_prompt = PromptTemplate.from_template("""
    Extract the animals mentioned in this text as a comma-separated list.
    Text: {text}
    """)
    
    # Prompt B: Creative Generation
    poem_prompt = PromptTemplate.from_template("""
    Write a 4-line poem about these animals: {animals}
    """)
    
    # LangChain LCEL (LangChain Expression Language) makes chaining easy.
    # We pipe the output of the first model call into the input variable of the second prompt.
    chain = extract_prompt | model | (lambda msg: {"animals": msg.content.strip()}) | poem_prompt | model
    
    result = chain.invoke({"text": text})
    logger.info(f"\nFinal Chained Output:\n{result.content.strip()}")


if __name__ == "__main__":
    demonstrate_openai_strategies()
    demonstrate_anthropic_strategies()
    demonstrate_google_strategies()
