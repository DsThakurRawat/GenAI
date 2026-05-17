"""
Module 01: Parametric vs. Non-Parametric Knowledge & Model Learning

This file explains the foundational concepts of how Large Language Models (LLMs) store knowledge,
how they learn, and contrasts updating weights via Fine-Tuning versus dynamic injection via RAG.

-------------------------------------------------------------------------------------------------
### 1. PARAMETRIC KNOWLEDGE
-------------------------------------------------------------------------------------------------
- Definition: Knowledge internalized directly within the model's static weights (parameters) 
  during pre-training and fine-tuning.
- How GPT / LLMs Learn Parametric Knowledge:
  1. Pre-Training: Self-supervised learning over massive web corpora. The model learns language 
     representations, grammar, reasoning capabilities, and world facts via Next-Token Prediction.
  2. Supervised Fine-Tuning (SFT): Adapting pre-trained weights using high-quality demonstration 
     pairs (Instruction, Response) to teach the model how to follow instructions and converse.
  3. Alignment (RLHF / DPO): Optimizing model outputs using human preferences for safety, helpfulness,
     and tone.
- Limitations: Static and frozen after training. Cannot answer queries about events occurring after 
  its knowledge cutoff date. Highly susceptible to hallucinations when forced to recall obscure facts.

-------------------------------------------------------------------------------------------------
### 2. NON-PARAMETRIC KNOWLEDGE
-------------------------------------------------------------------------------------------------
- Definition: Knowledge stored external to the model's weights, typically in databases, document 
  stores, or real-time APIs.
- Integration: Accessed dynamically at query time using Retrieval-Augmented Generation (RAG).
- Advantages: Always up-to-date, fully verifiable, zero risk of training-data contamination, and 
  does not require expensive weight retraining.

-------------------------------------------------------------------------------------------------
### 3. GETTING RECENT DATA VIA FINE-TUNING VS. RAG
-------------------------------------------------------------------------------------------------
- Can you use Fine-Tuning to teach an LLM recent factual data?
  Technically yes, but it is highly inefficient and strongly discouraged for pure fact storage:
  - Catastrophic Forgetting: Updating weights on new facts can corrupt or degrade pre-existing reasoning paths.
  - High Latency & Cost: Training requires GPU clusters and significant compute overhead.
  - Stale Knowledge: The moment fine-tuning completes, the knowledge is already static.
- Industry Standard Approach:
  Use Fine-Tuning to teach the model *Behavior, Structure, Tone, and Domain-Specific Tasks*.
  Use RAG to provide *Factual, Recent, and Private Data*.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Configure robust professional logging
logging.basicConfig(level=logging.INFO, format='\n%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Helper function to get model safely or mock if API keys are unconfigured placeholders
def get_chat_model():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.warning("[Notice]: Valid GOOGLE_API_KEY not found in .env. Running in Educational Simulation Mode.")
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    except Exception as e:
        logger.warning(f"[Notice]: Could not initialize real model ({str(e)}). Running in Educational Simulation Mode.")
        return None

def demonstrate_parametric_limitation(model):
    """
    Demonstrates querying the model relying solely on its frozen Parametric Knowledge.
    When asked about highly specific real-time or localized facts outside its training scope,
    it either refuses to answer or hallucinates.
    """
    logger.info("="*70)
    logger.info("1. DEMONSTRATING PARAMETRIC KNOWLEDGE LIMITATIONS")
    logger.info("="*70)

    question = "What is the exact status of the ACME Corp internal server migration scheduled for today?"
    logger.info(f"[Input Query]: {question}")

    if model:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant answering based on your internal parametric knowledge."),
                ("human", "{question}")
            ])
            response = (prompt | model).invoke({"question": question})
            logger.info(f"[Parametric Output]: {response.content.strip()}")
            return
        except Exception as e:
            logger.error(f"[API Error]: {str(e)}")
            
    # Simulated execution flow to ensure seamless local educational demonstration
    logger.info("[Simulated Output]: I'm sorry, as an AI model my knowledge cutoff is static. I do not have access to private company networks or events scheduled for today.")


def demonstrate_non_parametric_injection(model):
    """
    Demonstrates overcoming parametric limitations by combining static weights with 
    Non-Parametric Knowledge injected dynamically at query time (Core RAG concept).
    """
    logger.info("\n" + "="*70)
    logger.info("2. INJECTING NON-PARAMETRIC KNOWLEDGE (DYNAMIC RAG SIMULATION)")
    logger.info("="*70)

    simulated_retrieved_context = (
        "ACME Corp Internal Log - Server Migration Update:\n"
        "Timestamp: Today, 08:30 AM.\n"
        "Status: The migration of database shard-01 completed successfully. Shard-02 is delayed "
        "by 2 hours due to a transient network bottleneck. All critical customer-facing endpoints remain stable."
    )

    question = "What is the exact status of the ACME Corp internal server migration scheduled for today?"
    logger.info(f"[Injected Context]:\n{simulated_retrieved_context}\n")
    logger.info(f"[Input Query]: {question}")

    if model:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a precise corporate IT assistant. Answer the user's question using ONLY the "
                    "provided real-time context. Do not extrapolate or rely on external parameters."
                )),
                ("human", "Context:\n{context}\n\nQuestion: {question}")
            ])
            response = (prompt | model).invoke({
                "context": simulated_retrieved_context,
                "question": question
            })
            logger.info(f"[Augmented Output]: {response.content.strip()}")
            return
        except Exception as e:
            logger.error(f"[API Error]: {str(e)}")

    # Simulated execution flow
    logger.info("[Simulated Output]: Based on the internal logs from Today at 08:30 AM, database shard-01 migrated successfully, but shard-02 is experiencing a 2-hour delay due to a network bottleneck. All critical customer-facing endpoints remain stable.")


if __name__ == "__main__":
    llm = get_chat_model()
    demonstrate_parametric_limitation(llm)
    demonstrate_non_parametric_injection(llm)
