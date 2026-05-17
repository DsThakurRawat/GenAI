"""
Module 02: Fine-Tuning Deep Dive, Types, Techniques & Core Limitations

This file provides a comprehensive architectural breakdown of Fine-Tuning, detailing
the different techniques, formatting requirements, and critical industry limitations.

-------------------------------------------------------------------------------------------------
### 1. TYPES & TECHNIQUES OF FINE-TUNING
-------------------------------------------------------------------------------------------------
1. Full Fine-Tuning:
   - Updates all parameters/weights across all layers of the base model.
   - Extremely resource-intensive. Prone to catastrophic forgetting if the dataset is small or biased.

2. Parameter-Efficient Fine-Tuning (PEFT) / LoRA (Low-Rank Adaptation):
   - Freezes the base model weights and injects small, trainable low-rank decomposition matrices 
     into specific layers (typically attention layers).
   - Reduces trainable parameters by up to 99%, making fine-tuning possible on consumer GPUs.

3. Instruction Fine-Tuning:
   - Teaching the base completion model to act as a dialogue/instruction assistant.
   - Data format: Structured prompt/completion mappings (e.g., {"prompt": "Translate to French: Hello", "completion": "Bonjour"}).

4. Domain Adaptation (Continued Pre-Training):
   - Training the model on a massive corpus of raw text from a highly specific domain (Legal, Medical).
   - Teaches specialized vocabulary and statistical style distributions rather than specific task execution.

-------------------------------------------------------------------------------------------------
### 2. CORE PROBLEMS & LIMITATIONS OF FINE-TUNING
-------------------------------------------------------------------------------------------------
1. Stale Knowledge & Inability to Update Dynamically:
   - Fine-tuning creates a snapshot. Once training completes, any fact occurring the next second is missing.
   - Continuous retraining is highly unscalable and cost-prohibitive.

2. Catastrophic Forgetting:
   - As the model adjusts its weights to minimize loss on new domain data, it can unlearn or degrade 
     general capabilities, reasoning coherence, or alignment safety filters.

3. Total Lack of Source Auditability (Zero Citations):
   - When an LLM outputs a string from fine-tuned weights, it provides zero verifiable proof of where 
     it learned that fact. It cannot cite specific source files.

4. Extreme Compute Footprint:
   - Requires specialized pipeline orchestrations, evaluation test sets, gradient check-pointing, 
     and dedicated deployment endpoints.

-------------------------------------------------------------------------------------------------
### 3. EXECUTABLE DEMONSTRATION: PREPARING FINE-TUNING DATASETS
-------------------------------------------------------------------------------------------------
Below, we demonstrate creating structured instruction fine-tuning datasets using LangChain components.
"""

import json
import logging
from langchain_core.prompts import PromptTemplate

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='\n%(message)s')
logger = logging.getLogger(__name__)

def generate_finetuning_dataset_example():
    """
    Demonstrates generating fine-tuning formatting (JSONL format) using LangChain templates.
    Fine-tuning requires transforming raw unstructured text into structured System/User/Assistant pairs.
    """
    logger.info("="*70)
    logger.info("PREPARING INSTRUCTION TUNING DATASETS (OPENAI / GEMINI JSONL FORMAT)")
    logger.info("="*70)

    # Base prompt template used to ensure consistent training data generation
    instruction_template = PromptTemplate.from_template(
        "You are an expert customer sentiment classifier. Classify the user input into [POSITIVE, NEGATIVE, NEUTRAL].\n\nInput: {user_input}"
    )

    raw_training_data = [
        {"input": "The interface is incredibly intuitive and fast.", "label": "POSITIVE"},
        {"input": "It crashed immediately after loading my dashboard.", "label": "NEGATIVE"},
        {"input": "The app works as expected, nothing special.", "label": "NEUTRAL"}
    ]

    formatted_jsonl_records = []

    for entry in raw_training_data:
        # Generate the formatted system/user instruction prompt
        formatted_prompt = instruction_template.format(user_input=entry["input"])
        
        # Build standard chat-completion training payload
        training_record = {
            "messages": [
                {"role": "system", "content": "You are an expert customer sentiment classifier."},
                {"role": "user", "content": entry["input"]},
                {"role": "assistant", "content": entry["label"]}
            ]
        }
        formatted_jsonl_records.append(training_record)

    logger.info("[Sample JSONL Training Records Generated]:")
    for record in formatted_jsonl_records:
        logger.info(json.dumps(record, indent=2))

    logger.info("\n[Key Takeaway]:")
    logger.info("Fine-tuning teaches the model the *Format* and *Task Boundary* (outputting exactly POSITIVE/NEGATIVE/NEUTRAL).")
    logger.info("It should NOT be used to memorize real-time database entries.")


if __name__ == "__main__":
    generate_finetuning_dataset_example()
