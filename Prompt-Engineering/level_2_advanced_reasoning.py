"""
Level 2: Advanced Reasoning Techniques

This module covers techniques to overcome the "System 1" (fast, instinctual) nature of LLMs,
forcing them to use "System 2" (slow, deliberate) reasoning. 

Topics Covered:
1. Zero-Shot Chain of Thought (CoT)
2. Few-Shot Chain of Thought (CoT)
3. Self-Consistency (Majority Voting)
4. Directional Stimulus Prompting (Hints)
5. Generated Knowledge Prompting (Knowledge generation before answering)
6. Step-Back Prompting (Abstracting the problem first)
"""

import logging
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='\n%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# We use temperature=0.0 to ensure the most logical reasoning path.
# We use temperature=0.7 for Self-Consistency to ensure diverse reasoning paths.
model_deterministic = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
model_creative = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)


def demonstrate_zero_shot_cot():
    """
    1. Zero-Shot Chain of Thought (CoT)
    
    LLMs generate output token by token. If they jump straight to the answer, 
    they cannot backtrack if the math was wrong. By appending "Let's think step by step", 
    we force the model to output its reasoning tokens FIRST. The final answer token is then 
    conditioned on all the preceding correct reasoning tokens.
    """
    logger.info("="*50)
    logger.info("1. ZERO-SHOT CHAIN OF THOUGHT (CoT)")
    logger.info("="*50)
    
    math_problem = "A farmer has 100 chickens. 20 of them are roosters. Half of the hens lay 1 egg a day. The other half lay 2. How many eggs does he get in a week?"
    
    cot_prompt = PromptTemplate.from_template("""
    Question: {problem}
    
    Let's think step by step to find the answer. End your response with 'Final Answer: [number]'.
    """)
    
    logger.info(f"Problem: {math_problem}\n")
    result = (cot_prompt | model_deterministic).invoke({"problem": math_problem})
    logger.info(result.content.strip())


def demonstrate_few_shot_cot():
    """
    2. Few-Shot Chain of Thought (CoT)
    
    Sometimes "Let's think step by step" isn't enough, especially if the logic is unique 
    to your business domain. Few-Shot CoT provides explicit examples of *how* the model 
    should reason.
    """
    logger.info("\n" + "="*50)
    logger.info("2. FEW-SHOT CHAIN OF THOUGHT (CoT)")
    logger.info("="*50)
    
    few_shot_cot_prompt = PromptTemplate.from_template("""
    Example 1:
    Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
    A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
    
    Example 2:
    Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
    A: The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The answer is 9.
    
    Your Turn:
    Q: {problem}
    A: 
    """)
    
    new_problem = "Sarah has 5 boxes of pencils. Each box contains 10 pencils. She gives 15 pencils to her brother and then buys 2 more boxes. How many pencils does she have?"
    
    logger.info(f"Problem: {new_problem}\n")
    result = (few_shot_cot_prompt | model_deterministic).invoke({"problem": new_problem})
    logger.info(result.content.strip())


def demonstrate_self_consistency():
    """
    3. Self-Consistency (Majority Voting)
    
    Complex problems might have multiple valid reasoning paths, or the model might make 
    a random mistake. Self-Consistency solves this by asking the LLM the same question 
    multiple times (with a high temperature so the paths differ) and taking the most 
    common final answer.
    """
    logger.info("\n" + "="*50)
    logger.info("3. SELF-CONSISTENCY")
    logger.info("="*50)
    
    problem = "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?"
    prompt = PromptTemplate.from_template("Question: {problem}\nLet's think step by step. End with 'Final Answer: [number]'.")
    
    logger.info(f"Problem: {problem}\nGenerating 3 reasoning paths...")
    
    answers = []
    for i in range(3):
        # Notice we use the CREATIVE model here so we get diverse reasoning
        result = (prompt | model_creative).invoke({"problem": problem})
        # Extracting just the final answer for brevity
        final_line = result.content.strip().split('\n')[-1]
        answers.append(final_line)
        logger.info(f"Path {i+1} concluded: {final_line}")
        
    logger.info("\n(In production, you write Python code to parse these strings and return the most frequent one.)")


def demonstrate_generated_knowledge():
    """
    4. Generated Knowledge Prompting
    
    Sometimes a model hallucinates because it tries to answer a question before retrieving 
    the necessary facts from its weights. This technique uses two prompts:
    Prompt 1: Generate facts about the topic.
    Prompt 2: Answer the question using the generated facts.
    """
    logger.info("\n" + "="*50)
    logger.info("4. GENERATED KNOWLEDGE PROMPTING")
    logger.info("="*50)
    
    topic = "The James Webb Space Telescope vs Hubble"
    question = "Which telescope is better suited to see through dense dust clouds and why?"
    
    logger.info(f"Question: {question}\n")
    
    # Step 1: Generate Knowledge
    knowledge_prompt = PromptTemplate.from_template("Generate 3 key technical facts comparing {topic}.")
    knowledge = (knowledge_prompt | model_deterministic).invoke({"topic": topic}).content
    logger.info(f"--- Step 1: Generated Knowledge ---\n{knowledge.strip()}\n")
    
    # Step 2: Integrate into final answer
    integration_prompt = PromptTemplate.from_template("""
    Background Knowledge:
    {knowledge}
    
    Using the background knowledge provided, answer the following question.
    Question: {question}
    """)
    final_answer = (integration_prompt | model_deterministic).invoke({"knowledge": knowledge, "question": question}).content
    logger.info(f"--- Step 2: Final Answer ---\n{final_answer.strip()}")


if __name__ == "__main__":
    demonstrate_zero_shot_cot()
    demonstrate_few_shot_cot()
    demonstrate_self_consistency()
    demonstrate_generated_knowledge()
