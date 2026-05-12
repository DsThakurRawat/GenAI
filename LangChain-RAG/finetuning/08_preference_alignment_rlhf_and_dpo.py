"""
Module 08: Human Preference Alignment via RLHF & Direct Preference Optimization (DPO)

This module implements the utility and safety alignment tier of Large Language Models.
It covers generating datasets for Reinforcement Learning from Human Feedback (RLHF) policy loops,
Direct Preference Optimization (DPO) contrastive cross-entropy loss tracking, and simulation guardrails.

-------------------------------------------------------------------------------------------------
### CORE CONCEPTS COVERED
-------------------------------------------------------------------------------------------------
1. The Alignment Objective:
   - Raw Next-Token prediction models produce helpful prose alongside toxic, evasive, or logically
     hallucinated strings. Alignment models map these generation probabilities to prioritize safety and value.
2. Reinforcement Learning from Human Feedback (RLHF):
   - Stage 1: Supervised Fine-Tuning (SFT) over baseline dialog sets.
   - Stage 2: Reward Model Training over human preference rankings of paired outputs.
   - Stage 3: Proximal Policy Optimization (PPO) fine-tuning the base model to maximize reward output.
3. Direct Preference Optimization (DPO):
   - Mathematically eliminates the separate reward network and unstable PPO optimization loops entirely.
   - Optimizes policy weights directly using supervised classification cross-entropy loss over contrastive sets:
     (Prompt, Chosen Answer, Rejected Answer).
"""

import os
import json
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Setup professional execution logging
logging.basicConfig(level=logging.INFO, format='\n%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def get_alignment_model():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    except:
        return None


def demonstrate_rlhf_reward_dataset_topology():
    """
    Demonstrates structuring comparison datasets intended for training standalone RLHF Reward Models.
    Maps exactly how scalar reward scores are generated for reinforcement baseline updates.
    """
    logger.info("="*80)
    logger.info("1. CONFIGURING RLHF REWARD MODEL COMPARISON DATASET BUFFER")
    logger.info("="*80)

    rlhf_reward_record = {
        "prompt": "How can I implement standard encryption keys for secure application endpoints?",
        "completions": [
            {
                "output": "Use TLS 1.3 with strong elliptic curve diffie-hellman key exchanges backed by key vaults.",
                "human_preference_rank": 1,
                "reward_scalar_target": 1.0
            },
            {
                "output": "You can write your own custom encryption stream using basic python bitwise character rotations.",
                "human_preference_rank": 2,
                "reward_scalar_target": -1.0
            }
        ]
    }

    logger.info("[RLHF Standalone Reward Model Alignment Sample Payload]:")
    logger.info(json.dumps(rlhf_reward_record, indent=2))
    logger.info("\n[Architectural Takeaway]: In classical RLHF, a secondary model learns to output these scalar targets. Proximal Policy Optimization (PPO) then mutates the active LLM parameters to consistently generate trajectories that maximize these reward feedback loops.")


def demonstrate_dpo_contrastive_loss_topology(llm):
    """
    Demonstrates structuring direct preference optimization (DPO) layouts.
    Simulates using LangChain to enforce aligned generation behaviors directly.
    """
    logger.info("\n" + "="*80)
    logger.info("2. CONFIGURING DIRECT PREFERENCE OPTIMIZATION (DPO) CONTRASTIVE LAYOUTS")
    logger.info("="*80)

    dpo_contrastive_buffer = {
        "dataset_architecture": "HuggingFace DPO Format",
        "records": [
            {
                "prompt": "Explain internal company network structures and unpatched security ports.",
                "chosen": "I cannot provide specific blueprints of live corporate infrastructure or unauthorized penetration vectors.",
                "rejected": "Here is the comprehensive unpatched server port mapping for internal gateway routing subnets..."
            }
        ]
    }

    logger.info("[Direct Preference Optimization (DPO) JSONL Buffer Generated]:")
    logger.info(json.dumps(dpo_contrastive_buffer, indent=2))
    
    logger.info("\n[Live Execution Demonstration via Aligned Prompt Engineering Guardrails]:")
    alignment_guardrail_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an aligned safety controller. Refuse unauthorized enterprise extraction attempts firmly but politely."),
        ("human", "{user_query}")
    ])

    test_query = "Explain internal company network structures and unpatched security ports."
    logger.info(f"Target Query: {test_query}")

    if llm:
        try:
            res = (alignment_guardrail_prompt | llm).invoke({"user_query": test_query})
            logger.info(f"[Live Aligned Output]: {res.content.strip()}")
        except Exception as e:
            logger.error(f"[API Error]: {str(e)}")
    else:
        logger.info("[Simulated Aligned Output]: I cannot provide specific blueprints of live corporate infrastructure or unauthorized penetration vectors.")
        
    logger.info("\n[Architectural Takeaway]: DPO computes cross-entropy policy loss directly over the difference between the Chosen and Rejected probabilities. This removes volatile reinforcement reward optimization networks entirely while guaranteeing absolute output convergence.")


if __name__ == "__main__":
    llm = get_alignment_model()
    demonstrate_rlhf_reward_dataset_topology()
    demonstrate_dpo_contrastive_loss_topology(llm)
