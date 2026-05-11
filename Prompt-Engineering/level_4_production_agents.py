"""
Level 4: Prompting in Production (Agentic AI)

This module moves beyond simple Q&A and explores how Prompt Engineering forms the 
foundation of complex AI systems and Agents.

Topics Covered:
1. RAG (Retrieval-Augmented Generation) - Grounding models in external data.
2. ReAct (Reasoning and Acting) - The core prompt loop that enables AI Agents.
3. PAL (Program-Aided Language Models) - Using code execution to solve logic/math.
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='\n%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

def demonstrate_rag():
    """
    1. Retrieval-Augmented Generation (RAG)
    
    The biggest problem with LLMs is hallucinations (making things up confidently).
    RAG solves this by turning the LLM from a "closed-book test taker" into an 
    "open-book test taker."
    
    The Prompt Engineering aspect of RAG relies on STRICT GROUNDING INSTRUCTIONS.
    You must explicitly tell the model: "DO NOT use your pre-trained knowledge. ONLY use the context provided."
    """
    logger.info("="*50)
    logger.info("1. RAG (RETRIEVAL-AUGMENTED GENERATION)")
    logger.info("="*50)
    
    rag_prompt = PromptTemplate.from_template("""
    You are a strict, factual assistant. 
    
    INSTRUCTIONS:
    1. Answer the Question using ONLY the information provided in the Context.
    2. If the Context does not contain the answer, you must output exactly: "Information not found in database."
    3. Do not rely on any outside knowledge.
    
    CONTEXT:
    {context}
    
    QUESTION: {question}
    """)
    
    # Mocking the retrieval step (usually done via Vector Database like Pinecone/Chroma)
    mock_db_context = "Project Apollo launches on Q3 2024. The budget is $5M. The project lead is Sarah Connor."
    
    q1 = "What is the budget for Project Apollo?"
    logger.info(f"Question 1: {q1}")
    logger.info(f"Answer: {(rag_prompt | model).invoke({'context': mock_db_context, 'question': q1}).content.strip()}")
    
    q2 = "Who is the CEO of the company running Project Apollo?"
    logger.info(f"\nQuestion 2: {q2}")
    logger.info(f"Answer (Testing Fallback): {(rag_prompt | model).invoke({'context': mock_db_context, 'question': q2}).content.strip()}")


# --- Setup for ReAct Agent ---
@tool
def calculate_shipping_cost(weight_kg: float) -> str:
    """Returns the shipping cost based on the weight in kg. Rate is $5 per kg."""
    return f"The shipping cost is ${weight_kg * 5:.2f}"

def demonstrate_react():
    """
    2. ReAct (Reasoning and Acting)
    
    ReAct is the prompt framework that powers LangChain Agents. 
    It forces the model into a loop:
    - THOUGHT: What do I need to do?
    - ACTION: Which tool should I use and with what inputs?
    - OBSERVATION: What was the result of the tool?
    - THOUGHT: Do I have the answer now? (If no, loop again).
    
    We demonstrate this by giving the model a custom tool.
    """
    logger.info("\n" + "="*50)
    logger.info("2. REACT (REASONING AND ACTING) AGENT")
    logger.info("="*50)
    
    tools = [calculate_shipping_cost]
    
    # LangChain handles the exact ReAct string formatting under the hood when using create_tool_calling_agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful logistics assistant. You have access to tools to calculate costs. Use them!"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # This is where the Thought/Action/Observation log goes
    ])
    
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True shows the ReAct loop
    
    logger.info("Executing Agent Query (Watch the console for the Thought/Action loop)...")
    try:
        agent_executor.invoke({"input": "I have a package that weighs 12.5 kg. How much will it cost to ship?"})
    except Exception as e:
        logger.error(f"Agent failed: {e}")


def demonstrate_pal():
    """
    3. PAL (Program-Aided Language Models)
    
    Instead of using Chain-of-Thought for math (which LLMs are notoriously bad at, 
    even with CoT), PAL prompts the LLM to write a Python script to solve the problem. 
    
    The external runtime (Python) calculates the math perfectly, and the LLM 
    just reads the print statement.
    """
    logger.info("\n" + "="*50)
    logger.info("3. PAL (PROGRAM-AIDED LANGUAGE MODELS)")
    logger.info("="*50)
    
    pal_prompt = PromptTemplate.from_template("""
    You are a Python programmer. 
    Write a python script to solve the following math problem.
    Do not explain, just output the python code inside ```python blocks.
    The code must print the final answer.
    
    Problem: {problem}
    """)
    
    problem = "A financial portfolio starts at $10,000. It grows by 5% the first year, loses 2% the second year, and grows by 8% the third year. What is the final value?"
    
    logger.info(f"Problem: {problem}\n")
    logger.info("Prompting LLM to write Python code instead of doing the math itself...")
    
    result = (pal_prompt | model).invoke({"problem": problem})
    
    logger.info("Generated Python Code:")
    logger.info(result.content.strip())
    logger.info("\n(In a full PAL system, you would execute this code using Python's `exec()` or a sandbox environment to get the 100% accurate mathematical answer).")


if __name__ == "__main__":
    demonstrate_rag()
    demonstrate_react()
    demonstrate_pal()
