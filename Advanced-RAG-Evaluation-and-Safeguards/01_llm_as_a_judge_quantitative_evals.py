from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ============================================================================
# 01. QUANTITATIVE EVALUATIONS: LLM-AS-A-JUDGE SCORING PIPELINES
# ============================================================================
# This script demonstrates running automated eval loops using structured outputs
# to score standard RAG generation parameters for Faithfulness and Groundedness.

# --- 1. Define Explicit Structured Evaluation Scorecard Schema ---
class RagasEvaluationMetric(BaseModel):
    score_metric: Literal["PASS", "FAIL"] = Field(
        description="Select PASS strictly if the generation string maps completely to the provided context facts."
    )
    grounding_confidence: float = Field(
        ge=0.0, le=1.0, 
        description="Statistical confidence metrics for evaluation accuracy."
    )
    discrepancy_explanation: str = Field(
        description="Comprehensive cognitive audit tracing unverified external assumptions."
    )

# --- 2. Initialize the Judge Evaluation Engine ---
evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Wrap model to force typed outputs matching evaluation metrics blueprints
structured_judge = evaluator_llm.with_structured_output(RagasEvaluationMetric)

# --- 3. Compile Master Judge Prompt Template ---
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert quantitative AI model evaluator. 
    Compare the provided Context facts against the generated Candidate response.
    Output a structured scorecard rating if the Candidate response introduces unverified details."""),
    ("human", """
    [Source Context Facts]:
    {context}
    
    [Generated Candidate Text]:
    {candidate}
    """)
])

eval_chain = eval_prompt | structured_judge

# --- 4. Execution Pipeline Test Runs ---
if __name__ == "__main__":
    print("⚖️ Booting Quantitative LLM-as-a-Judge Evaluation Pipeline...")
    
    test_context = "The primary server connection protocol times out strictly after 30 seconds under heavy load."
    
    # 1. Test Case A: Perfectly Grounded Generation String
    print("\n[Audit Run 1]: Evaluating perfectly faithful output strings...")
    faithful_candidate = "Server connections terminate at 30 seconds during load spikes."
    
    score_a: RagasEvaluationMetric = eval_chain.invoke({
        "context": test_context,
        "candidate": faithful_candidate
    })
    
    print(f"-> Verification Output Status: {score_a.score_metric}")
    print(f"-> Evaluator Confidence: {score_a.grounding_confidence}")
    print(f"-> Discrepancy Analysis: {score_a.discrepancy_explanation}")
    
    # 2. Test Case B: Unverified Hallucinatory Extrapolation String
    print("\n[Audit Run 2]: Evaluating candidate string introducing hallucinated constraints...")
    hallucinated_candidate = "Connections terminate after 30 seconds. To prevent this, increase the cache threshold to 1024MB."
    
    score_b: RagasEvaluationMetric = eval_chain.invoke({
        "context": test_context,
        "candidate": hallucinated_candidate
    })
    
    print(f"-> Verification Output Status: {score_b.score_metric}")
    print(f"-> Evaluator Confidence: {score_b.grounding_confidence}")
    print(f"-> Discrepancy Analysis: {score_b.discrepancy_explanation}")
    
    print("\n✅ Automated Evals Architecture Script Executed Cleanly.")
