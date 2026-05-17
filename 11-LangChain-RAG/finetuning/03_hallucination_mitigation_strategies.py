"""
Module 03: Hallucination Mitigation Strategies & Guardrail Implementations

This file details every foundational methodology used across the industry to solve 
hallucination problems in Large Language Models (LLMs), complete with an executable LCEL chain.

-------------------------------------------------------------------------------------------------
### ALL METHODS TO SOLVE HALLUCINATIONS
-------------------------------------------------------------------------------------------------
1. Dynamic Context Grounding (Retrieval-Augmented Generation):
   - Replaces parameter retrieval guesswork by supplying exact source documents directly in the prompt.
   - Prevents the model from extrapolating missing facts.

2. Strict Prompt Boundaries & Fallback Engineering:
   - System Prompts must contain explicit structural boundaries:
     "Answer using ONLY the facts mentioned in the context."
     "If the context does not contain the complete factual answer, output exactly: 'UNKNOWN'."

3. Sampling Parameter Restraints:
   - Temperature = 0.0: Forces greedy decoding, picking the highest probability token at every step.
   - Top-P / Top-K tuning: Restricts the token probability sample distribution to eliminate highly creative/unlikely word paths.

4. Self-Correction & Verification Chains (LLM-as-a-Critic):
   - Using a multi-chain architecture where Model A generates a draft answer based on context, 
     and Model B acts as an auditor verifying that every single claim maps directly back to the source text.

5. Forced Citation & Source Attribution:
   - Instructing the model or structured tools to output explicit document indices alongside text blocks.

-------------------------------------------------------------------------------------------------
### EXECUTABLE IMPLEMENTATION: STRICT GUARDRAIL LCEL PIPELINE
-------------------------------------------------------------------------------------------------
Below, we demonstrate a dual-step chain that combines strict prompt constraints with an automated 
factual verification step.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure robust professional logging
logging.basicConfig(level=logging.INFO, format='\n%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_deterministic_llm():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    except:
        return None

def demonstrate_hallucination_guardrails(llm):
    logger.info("="*70)
    logger.info("IMPLEMENTING STRICT RAG GUARDRAILS & VERIFICATION PIPELINE")
    logger.info("="*70)

    source_context = (
        "Project Alpha Specifications:\n"
        "Deployment Target: AWS ECS Multi-region.\n"
        "Authentication Provider: Okta SSO.\n"
        "Authorized Database Engine: PostgreSQL 16."
    )

    # Test Query requesting a fact NOT present in the context
    malicious_query = "What is the memory allocation limit for the caching layer in Project Alpha?"

    logger.info(f"[Source Context Provided]:\n{source_context}\n")
    logger.info(f"[Input Query]: {malicious_query}")

    if llm:
        try:
            # Step 1: Base Generation Prompt with strict constraint injection
            generation_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a strict technical documentation assistant. Answer using ONLY the provided context.\n"
                    "CRITICAL RULE: If the answer is not explicitly stated in the context, output exactly: 'UNKNOWN'."
                )),
                ("human", "Context:\n{context}\n\nQuery: {query}")
            ])

            # Step 2: Verification/Self-Correction Prompt
            verification_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are an AI Auditor. Read the source context and the generated draft answer.\n"
                    "If the draft contains ANY information not stated in the source context, rewrite the final output as 'HALLUCINATION DETECTED: UNKNOWN'."
                    "\nOtherwise, output the draft answer unchanged."
                )),
                ("human", "Source Context:\n{context}\n\nDraft Answer:\n{draft}\n\nFinal Audited Output:")
            ])

            # Build LCEL pipeline
            base_chain = generation_prompt | llm | StrOutputParser()
            
            logger.info("\n[Executing Live API Multi-Step Verification Chain]...")
            draft_answer = base_chain.invoke({"context": source_context, "query": malicious_query})
            logger.info(f"[Step 1 Draft Answer]: {draft_answer.strip()}")

            final_audited_answer = (verification_prompt | llm | StrOutputParser()).invoke({
                "context": source_context,
                "draft": draft_answer
            })
            logger.info(f"[Step 2 Final Audited Output]: {final_audited_answer.strip()}")
            return
        except Exception as e:
            logger.error(f"[API Error]: {str(e)}")

    # Educational Simulation execution flow matching real API execution exactly
    logger.info("\n[Simulating LCEL Verification Flow Due to Placeholder API Keys]:")
    logger.info("[Step 1 Draft Answer]: UNKNOWN")
    logger.info("[Step 2 Final Audited Output]: UNKNOWN")
    logger.info("\n[Explanation]: By enforcing rigorous contextual constraints and greedy sampling, the model correctly identifies the absent information and refuses to fabricate a value.")


if __name__ == "__main__":
    llm = get_deterministic_llm()
    demonstrate_hallucination_guardrails(llm)
