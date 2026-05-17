from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field
from src.core.logger import logger

# CONCEPTUAL: Local LLMs & Quantization
"""
Running models locally (on your own hardware) is great for privacy and cost.
However, LLMs are huge. To fit them on a normal GPU/CPU, we use 'Quantization'.

1. What is Quantization?: It's like compressing a 4K video to 1080p. 
   We reduce the precision of the model's weights (e.g., from 16-bit to 4-bit).
2. Why use it?: It allows a 7 Billion parameter model to run on 8GB of RAM instead of 28GB.
3. HuggingFacePipeline: LangChain's way to wrap local 'transformers' models.
"""

"""
LOCAL HUGGINGFACE MODEL DEMONSTRATION
=====================================

This script demonstrates how to run LLMs locally on your own machine using 
Transformers and LangChain's `HuggingFacePipeline`.

### PARAMETER IMPACT GUIDE (Docstring Explanation):

1. **model_id (str)**:
   - *What it is*: The local path or HF Hub ID of the model weights.
   - *Variation*: `TinyLlama-1.1B` (Small/Fast) vs `Llama-3-8B` (Large/Smart).

2. **max_new_tokens (int)**:
   - *Impact*: Controls the number of tokens the model generates. 
   - *Decision*: Set higher for stories, lower for classification to save VRAM/Time.

3. **device_map (str)**:
   - *Impact*: `auto` will try to use your GPU (CUDA) first, then CPU.
   - *Importance*: Using a GPU is significantly faster (10x-50x) than CPU for LLMs.

4. **temperature (float)**:
   - *Impact*: Controls randomness. In local pipelines, this is often passed in `pipeline_kwargs`.
"""

# Define a Pydantic model for our local task
class CodeAnalysis(BaseModel):
    language: str = Field(description="The programming language of the code")
    is_buggy: bool = Field(description="Whether the code has obvious bugs")
    suggestion: str = Field(description="A 1-sentence suggestion for improvement")

def run_local_model_demo():
    try:
        # Note: Running this will download a model (can be 1GB to 15GB+)
        # For a light demo, we use a very small model 'gpt2' or 'TinyLlama'
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        logger.info(f"Loading local model: {model_id}...")
        
        # 1. Load Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
      
        # 2. Create a Transformers Pipeline
        pipe = pipeline(
            "text-generation", 
            
            # PARAMETER: model
            # IMPACT: The actual neural network weights loaded into RAM/VRAM.
            model=model, 
            
            # PARAMETER: tokenizer
            # IMPACT: Converts your text to numbers the model understands.
            tokenizer=tokenizer, 
            
            # PARAMETER: max_new_tokens
            # IMPACT: Limits the length of the AI's response.
            max_new_tokens=100,
            
            # PARAMETER: device_map
            # IMPACT: 'auto' determines if the model runs on CPU or GPU.
            device_map="auto" 
        )

        # 3. Wrap in LangChain
        hf = HuggingFacePipeline(pipeline=pipe)

        logger.success(f"Local model {model_id} loaded successfully.")

        # Test the model
        code = "def add(a, b): return a + b"
        logger.info(f"Analyzing code locally: {code}")
        
        response = hf.invoke(f"Analyze this python code: {code}")
        
        print(f"\n--- Local Model Output ---\n{response}")

    except Exception as e:
        logger.error(f"Local Model Error: {e}")
        print("\n[TIP] Local models require significant RAM/GPU and disk space.")

if __name__ == "__main__":
    run_local_model_demo()
