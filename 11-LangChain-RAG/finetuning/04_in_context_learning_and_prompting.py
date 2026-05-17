"""
Module 04: In-Context Learning, Prompting Paradigms & Emergent LLM Properties

This file covers how foundation models learn dynamically at runtime without weight updates,
explores prompting paradigms, and examines emergent properties as models scale.

-------------------------------------------------------------------------------------------------
### 1. IN-CONTEXT LEARNING (ICL) & HOW GPT LEARNS DYNAMICALLY
-------------------------------------------------------------------------------------------------
- Definition: A paradigm where an LLM infers how to perform a task purely by analyzing context and 
  demonstration examples provided in its prompt at inference time.
- Weight Status: ZERO parameter updates occur during In-Context Learning. The internal weights 
  remain entirely frozen. The model relies on internal attention heads mapping the pattern seen 
  in the prompt context.
- Contrast with Fine-Tuning: While Fine-Tuning permanently alters weight matrices using backpropagation,
  In-Context Learning acts as transient instruction tracking inside the active context window.

-------------------------------------------------------------------------------------------------
### 2. ALL TYPES OF PROMPTING & LEARNING PARADIGMS
-------------------------------------------------------------------------------------------------
1. Zero-Shot Prompting:
   - Requesting task completion directly without providing reference examples.
   - Relies completely on pre-trained baseline capabilities.

2. Few-Shot Prompting (Language Models are Few-Shot Learners):
   - Providing 1 to 5 highly curated Input/Output mapping pairs.
   - Extremely effective at enforcing non-standard syntax, exact tone matching, and JSON/XML generation.

3. Chain of Thought (CoT) Prompting:
   - Forcing the model to output intermediate rationales ("Let's think step by step") before 
     stating the final answer. Allocates compute tokens to the intermediate reasoning phase.

-------------------------------------------------------------------------------------------------
### 3. PROPERTIES OF LLMs & EMERGENT PROPERTIES
-------------------------------------------------------------------------------------------------
- Foundational Properties: Autoregressive token generation, context window thresholds, static 
  knowledge cutoffs.
- Emergent Properties:
  - Definition: Capabilities or complex behaviors that appear unexpectedly only when a model exceeds 
    critical scale thresholds (parameters, training compute, depth).
  - Key Examples: 
    1. Few-Shot In-Context Learning (Small models fail to generalize from prompt examples; large models adapt flawlessly).
    2. Multi-step logical and spatial reasoning.
    3. Spontaneous translation across obscure language pairs without direct cross-training pairs.

-------------------------------------------------------------------------------------------------
### 4. CURRENT DEVELOPMENTS IN FOUNDATION MODELS
-------------------------------------------------------------------------------------------------
1. Test-Time Compute (Reasoning Models): Models like OpenAI o1 use hidden reinforcement-learning 
   reasoning loops to generate hundreds of internal CoT steps before outputting text.
2. Agentic ReAct Frameworks: Shifting from single prompt calls to multi-turn tool-use loops.
3. Native Multimodality: Vision, Audio, and Text processed via unified embedding tokenizers.

-------------------------------------------------------------------------------------------------
### EXECUTABLE IMPLEMENTATION: FEW-SHOT IN-CONTEXT LEARNING
-------------------------------------------------------------------------------------------------
Below, we demonstrate teaching the model a custom non-standard classification format via few-shot examples.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='\n%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def get_chat_model():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    except:
        return None

def demonstrate_few_shot_learning(llm):
    logger.info("="*70)
    logger.info("DEMONSTRATING IN-CONTEXT FEW-SHOT LEARNING VIA LANGCHAIN")
    logger.info("="*70)

    # Define strict input/output training examples demonstrating a unique output syntax
    examples = [
        {
            "input": "User clicked checkout but closed the tab.", 
            "output": "[ACTION: ABANDON] | [PRIORITY: P2] | [RE-TARGET: YES]"
        },
        {
            "input": "User successfully updated billing details.", 
            "output": "[ACTION: SUCCESS] | [PRIORITY: P4] | [RE-TARGET: NO]"
        }
    ]

    # Create the template for formatting individual examples
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])

    # Build the FewShot prompt template
    few_shot_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    # Assemble final master prompt template
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze user event triggers and output the precise status string format shown in examples."),
        few_shot_template,
        ("human", "{user_event}")
    ])

    test_input = "User added three items to cart, encountered a payment gateway timeout error."
    logger.info("[Few-Shot Demonstrations Injected]:")
    for ex in examples:
        logger.info(f"  Input:  {ex['input']}")
        logger.info(f"  Output: {ex['output']}\n")

    logger.info(f"[Live Evaluation Input]: {test_input}")

    if llm:
        try:
            chain = final_prompt | llm
            response = chain.invoke({"user_event": test_input})
            logger.info(f"[Emergent In-Context Output]: {response.content.strip()}")
            return
        except Exception as e:
            logger.error(f"[API Error]: {str(e)}")

    # Educational Simulation execution flow
    logger.info("\n[Simulating In-Context Inference Output]:")
    logger.info("[Emergent In-Context Output]: [ACTION: ERROR] | [PRIORITY: P1] | [RE-TARGET: YES]")
    logger.info("\n[Explanation]: The model infers the exact custom string syntax and domain logic dynamically from the embedded few-shot examples without executing any permanent weight updates.")


if __name__ == "__main__":
    llm = get_chat_model()
    demonstrate_few_shot_learning(llm)
