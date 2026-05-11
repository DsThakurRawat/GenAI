from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.core.logger import logger
from src.core.config import settings

"""
ANTHROPIC CHAT MODEL DEMONSTRATION
==================================

This script demonstrates how to initialize and use Anthropic's Claude models.
Claude is known for its high ethical standards and superior reasoning.

### PARAMETER IMPACT GUIDE (Docstring Explanation):

1. **model**: 
   - `claude-3-5-sonnet-20240620`: Balanced speed and intelligence.
   - `claude-3-opus-20240229`: Most powerful for complex research and coding.

2. **temperature (0.0 to 1.0)**:
   - Anthropic recommends a range between 0 and 1.
   - **Variation**: Use 0.0 for analytical tasks and 1.0 for creative brainstorming.

3. **max_tokens**:
   - **Mandatory**: Unlike some other providers, Anthropic often requires this value.
   - **Impact**: Limits the cost and ensures the model doesn't generate excessive output.
"""

# Initialize Anthropic Chat Model (Claude)
chat_model = ChatAnthropic(
    api_key=settings.ANTHROPIC_API_KEY,
    
    # PARAMETER: model
    # IMPACT: claude-3-5-sonnet is currently faster and smarter than many gpt-4 variants.
    model="claude-3-5-sonnet-20240620",
    
    # PARAMETER: temperature
    # IMPACT: 0.0 for strict logic; 0.7 for standard conversational depth.
    temperature=0.7,
    
    # PARAMETER: max_tokens
    # IMPACT: Controls the maximum response length.
    max_tokens=1024,
    
    # PARAMETER: streaming
    # IMPACT: Enables real-time token delivery to the user.
    streaming=True
)

# CONCEPTUAL: The Message Schema
"""
In production ChatModels, we don't just send a string. We send a list of Message Objects.
This is the core concept of 'Chat History' and 'Personas'.

1. SystemMessage: Sets the behavior/persona of the AI (The "Rules").
2. HumanMessage: What the user says.
3. AIMessage: What the AI previously said (used to provide context for follow-up questions).
"""

from pydantic import BaseModel, Field

# CONCEPTUAL: Structured Output with Claude
# Anthropic models are excellent at following Pydantic schemas.
class SecurityReport(BaseModel):
    vulnerability_found: bool = Field(description="Whether a security risk was detected")
    risk_level: str = Field(description="Low, Medium, or High")
    explanation: str = Field(description="Detailed explanation of the risk")

# Create a structured version of Claude
structured_claude = chat_model.with_structured_output(SecurityReport)

def run_anthropic_demo():
    try:
        code_snippet = "a = input(); eval(a)"
        logger.info(f"Analyzing Code Security: {code_snippet}")

        # This will return a SecurityReport object
        report = structured_claude.invoke(f"Audit this code: {code_snippet}")
        
        logger.success("Security analysis complete.")
        print(f"\n--- Security Report ---")
        print(f"Risk Found: {report.vulnerability_found}")
        print(f"Level: {report.risk_level}")
        print(f"Details: {report.explanation}")

    except Exception as e:
        logger.error(f"Anthropic API Error: {e}")
        print("\n[TIP] Ensure ANTHROPIC_API_KEY is set in your .env file.")

if __name__ == "__main__":
    run_anthropic_demo()
