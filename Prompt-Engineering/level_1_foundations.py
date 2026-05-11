"""
Level 1: Foundations of Prompt Design

This module covers the absolute fundamentals of Prompt Engineering. 
It goes beyond basic definitions, providing concrete examples of how slight 
tweaks in foundational prompts drastically alter the LLM's output.

Topics Covered:
1. The Anatomy of a Prompt (Instruction, Context, Input, Format)
2. Zero-Shot Prompting
3. Few-Shot Prompting (and the importance of example selection)
4. Role Prompting / Persona Adoption
5. Formatting Constraints (Enforcing structured output)
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='\n%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# We use temperature=0.0 to ensure deterministic, reproducible results for our prompt tests.
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

def demonstrate_anatomy():
    """
    1. The Anatomy of a Prompt
    
    A professional-grade prompt rarely consists of just a single sentence. 
    It is usually broken down into 4 distinct components:
    
    A. Instruction: The specific task (e.g., "Summarize", "Translate", "Extract").
    B. Context: The background information the model needs to understand the instruction.
    C. Input Data: The specific user text to process.
    D. Output Indicator: The exact format the model should use to reply.
    
    By separating these, the model doesn't confuse the instructions with the input data.
    """
    logger.info("="*50)
    logger.info("1. THE ANATOMY OF A PROMPT")
    logger.info("="*50)
    
    prompt = PromptTemplate.from_template("""
    ### INSTRUCTION ###
    Extract the names of all software tools mentioned in the text.
    
    ### CONTEXT ###
    You are an IT auditor reviewing a company's internal wiki to catalog their tech stack. 
    Ignore generic terms like "computer" or "server". Only extract specific software brands or names.
    
    ### INPUT DATA ###
    Text: {input_text}
    
    ### OUTPUT INDICATOR ###
    Return the result as a comma-separated list. If none are found, output "NONE".
    """)
    
    text = "Our engineers use VS Code for development and deploy to AWS. We communicate via Slack and manage tasks in Jira."
    
    logger.info(f"Input Text: {text}")
    result = (prompt | model).invoke({"input_text": text})
    logger.info(f"Output:\n{result.content.strip()}")


def demonstrate_zero_shot():
    """
    2. Zero-Shot Prompting
    
    "Zero-Shot" means providing ZERO examples of the expected output. You rely entirely 
    on the LLM's vast pre-trained knowledge base. 
    
    It works great for common tasks (translation, summarization), but often fails when 
    you need a very specific, non-standard output format.
    """
    logger.info("\n" + "="*50)
    logger.info("2. ZERO-SHOT PROMPTING")
    logger.info("="*50)
    
    prompt = PromptTemplate.from_template("""
    Classify the following customer support ticket into a department: 
    [Billing, Tech Support, Sales, General Inquiry].
    
    Ticket: {ticket}
    """)
    
    ticket = "My screen is flickering when I open the dashboard."
    
    logger.info(f"Ticket: {ticket}")
    result = (prompt | model).invoke({"ticket": ticket})
    logger.info(f"Output:\n{result.content.strip()}")


def demonstrate_few_shot():
    """
    3. Few-Shot Prompting (In-Context Learning)
    
    When Zero-Shot fails, Few-Shot is the answer. By providing 1 to 5 examples 
    (input/output pairs), the model learns the *pattern* and *format* you want.
    
    Crucial Rule: The examples you provide dictate the output. If your examples 
    are sloppy, the output will be sloppy.
    """
    logger.info("\n" + "="*50)
    logger.info("3. FEW-SHOT PROMPTING")
    logger.info("="*50)
    
    # We want a very specific output format that zero-shot would struggle to invent consistently.
    examples = [
        {"input": "I hate this new update. It crashes constantly.", "output": '{"severity": "HIGH", "sentiment": "NEGATIVE", "topic": "CRASH"}'},
        {"input": "How much does the enterprise plan cost?", "output": '{"severity": "LOW", "sentiment": "NEUTRAL", "topic": "PRICING"}'},
    ]
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the user input and return a JSON object with severity, sentiment, and topic."),
        few_shot_template,
        ("human", "{user_input}")
    ])
    
    test_input = "I can't log into my account, it keeps saying invalid password even though I just reset it."
    logger.info(f"Input: {test_input}")
    result = (final_prompt | model).invoke({"user_input": test_input})
    logger.info(f"Output (Notice how it perfectly matches the JSON format of the examples):\n{result.content.strip()}")


def demonstrate_role_prompting():
    """
    4. Role Prompting / Persona Adoption
    
    Telling the LLM to "Act as X" works because LLMs predict the next word based on 
    probabilities. By setting a persona, you shift the probability space. If you say 
    "Act as a Pirate", the probability of the word "Ahoy" increases drastically.
    If you say "Act as a Data Scientist", the probability of words like "variance" and "distribution" increases.
    """
    logger.info("\n" + "="*50)
    logger.info("4. ROLE PROMPTING")
    logger.info("="*50)
    
    topic = "Black Holes"
    
    # Persona 1: Kindergarten Teacher
    prompt_kindergarten = PromptTemplate.from_template(
        "Act as an enthusiastic kindergarten teacher. Explain {topic} to your class in 2 short sentences."
    )
    
    # Persona 2: Astrophysicist
    prompt_astrophysicist = PromptTemplate.from_template(
        "Act as a post-doctoral astrophysicist writing an abstract. Explain {topic} in 2 short sentences using high-level terminology."
    )
    
    logger.info("--- Persona 1: Kindergarten Teacher ---")
    res1 = (prompt_kindergarten | model).invoke({"topic": topic})
    logger.info(res1.content.strip())
    
    logger.info("\n--- Persona 2: Astrophysicist ---")
    res2 = (prompt_astrophysicist | model).invoke({"topic": topic})
    logger.info(res2.content.strip())


if __name__ == "__main__":
    demonstrate_anatomy()
    demonstrate_zero_shot()
    demonstrate_few_shot()
    demonstrate_role_prompting()
