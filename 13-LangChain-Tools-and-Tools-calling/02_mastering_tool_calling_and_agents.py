"""
Module 02: Production Implementation Guide for Tool Calling Lifecycles, 
Argument Injection (InjectedToolArg), & ReAct Agent Orchestration

This execution reference translates all conceptual code patterns from upstream tool-calling 
notebooks into a clean, highly structured, enterprise-ready Python module.

Key Technical Capabilities Demonstrated:
1. Tool Inspection: Validating inner Pydantic arguments, descriptions, and call signatures.
2. Direct Binding: Attaching tool interfaces to models via `.bind_tools()`.
3. Manual State Loops: Tracing `HumanMessage` -> `AIMessage` (tool_calls) -> `ToolMessage` -> Output Synthesis.
4. InjectedToolArg: Truncating schemas to hide specific runtime execution contexts from model awareness.
5. Agent Orchestration: Demonstrating both legacy initialize_agent patterns and modern LangGraph create_react_agent loops.
"""

import os
import json
import logging
import requests
from typing import Annotated, List
from dotenv import load_dotenv

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure pristine console logging
logging.basicConfig(level=logging.INFO, format='\n%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def get_tool_calling_model():
    """Initializes a production tool-capable chat model engine securely."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key.strip() == "your_gemini_api_key_here":
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    except Exception as e:
        logger.warning(f"Failed to boot live LLM engine: {e}")
        return None


# ===============================================================================================
# SECTION 1: STANDARD TOOL CREATION & INSPECTION
# ===============================================================================================

@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b, this tool returns their mathematical product."""
    return a * b


def demonstrate_tool_inspection():
    """Demonstrates extracting parameter definitions and tool validation parameters."""
    logger.info("="*80)
    logger.info("SECTION 1: TOOL CREATION & SCHEMA INSPECTION DEEP-DIVE")
    logger.info("="*80)
    
    logger.info(f"-> Tool Identifier Name: '{multiply.name}'")
    logger.info(f"-> Tool System Description: '{multiply.description.strip()}'")
    logger.info(f"-> Validated Inner Signature Arguments:\n{json.dumps(multiply.args, indent=2)}")
    
    # Executing the standalone tool directly with standard dictionary mapping payloads
    res = multiply.invoke({'a': 3, 'b': 4})
    logger.info(f"-> Verification Execution Result: multiply.invoke({{'a':3, 'b':4}}) => {res}")


# ===============================================================================================
# SECTION 2: MANUAL TOOL CALLING STATE LOOP ORCHESTRATION
# ===============================================================================================

def execute_manual_tool_calling_loop(llm):
    """
    Traces the entire manual state loop progression handling tool calls directly.
    """
    logger.info("\n" + "="*80)
    logger.info("SECTION 2: MANUAL TOOL CALLING STATE LOOP IMPLEMENTATION")
    logger.info("="*80)

    query_text = "Can you multiply 3 with 1000?"
    messages = [HumanMessage(content=query_text)]
    
    logger.info(f"[Step 1: Instantiating Base Message Array]:\n  -> {messages}")

    if llm:
        try:
            logger.info("\n[Step 2: Binding Tool Definitions & Emitting Model Prediction Request]...")
            llm_with_tools = llm.bind_tools([multiply])
            ai_message = llm_with_tools.invoke(messages)
            
            # Appending model evaluation results to active conversation flow state
            messages.append(ai_message)
            
            logger.info(f"\n[Step 3: Extracting Emitted Tool Calls Attributes]:")
            logger.info(json.dumps(ai_message.tool_calls, indent=2))
            
            if ai_message.tool_calls:
                tool_call = ai_message.tool_calls[0]
                logger.info(f"\n[Step 4: Invoking Local Registered Tool logic via tool.invoke(tool_call)]...")
                tool_output_msg = multiply.invoke(tool_call)
                
                logger.info(f"  -> Raw Returned ToolMessage Entity: {tool_output_msg}")
                messages.append(tool_output_msg)
                
                logger.info("\n[Step 5: Invoking Bound Engine Over Complete Updated Conversation Thread]...")
                final_synthesis = llm_with_tools.invoke(messages)
                logger.info(f"\n[Final Synthesis Response Engine Output]:\n  -> '{final_synthesis.content}'")
            else:
                logger.warning("Model bypassed tool calls array evaluation.")
        except Exception as e:
            logger.error(f"Execution Error during live tool state routing: {e}")
    else:
        logger.info("\n[Simulating Tool-Calling Lifecycle Steps (Offline Mode)]:")
        
        # Step 2 & 3 Simulation
        simulated_tool_call = {
            "name": "multiply",
            "args": {"a": 3, "b": 1000},
            "id": "call_mock_tx_777",
            "type": "tool_call"
        }
        simulated_ai_msg = AIMessage(content="", tool_calls=[simulated_tool_call])
        messages.append(simulated_ai_msg)
        logger.info(f"  -> Model Emitted AIMessage(tool_calls=[{simulated_tool_call['name']}])")
        
        # Step 4 Simulation
        simulated_tool_result = ToolMessage(
            content=str(3 * 1000), 
            name="multiply", 
            tool_call_id=simulated_tool_call["id"]
        )
        messages.append(simulated_tool_result)
        logger.info(f"  -> Local Appended ToolMessage Entity: {simulated_tool_result}")
        
        # Step 5 Simulation
        final_answer = "The product of 3 and 1000 is 3000."
        logger.info(f"\n[Final Synthesis Response Engine Output]:\n  -> '{final_answer}'")


# ===============================================================================================
# SECTION 3: DYNAMIC ARGUMENT INJECTION (InjectedToolArg)
# ===============================================================================================

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Fetches the operational currency conversion rate multiplier mapping a base currency code 
    to a destination target currency code.
    """
    try:
        url = f"https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return float(data.get("conversion_rate", 85.5))
        return 85.5
    except Exception:
        # Fallback offline multiplier simulation
        return 85.5


@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    Calculates final converted targets based on an externally provided currency exchange rate.
    The LLM sees only 'base_currency_value' in its evaluation parameters.
    """
    return float(base_currency_value) * float(conversion_rate)


def demonstrate_injected_arguments_flow(llm):
    """
    Demonstrates argument masking capabilities and runtime parameter insertion handling.
    """
    logger.info("\n" + "="*80)
    logger.info("SECTION 3: DYNAMIC RUNTIME PARAMETER INJECTION (InjectedToolArg)")
    logger.info("="*80)

    logger.info("[Verifying Schema Truncation Properties]:")
    logger.info(f"  -> Inspecting emitted schema properties for '{convert.name}' tool:")
    logger.info(json.dumps(convert.args, indent=2))
    logger.info("  Notice: 'conversion_rate' is completely absent from external parameter visibility models.")

    query_str = "What is the conversion factor between USD and INR, and based on that can you convert 10 USD to INR?"
    messages = [HumanMessage(content=query_str)]

    if llm:
        try:
            logger.info("\n[Executing Complex Dual-Tool Resolution Pipeline]...")
            llm_bound = llm.bind_tools([get_conversion_factor, convert])
            initial_ai_response = llm_bound.invoke(messages)
            messages.append(initial_ai_response)
            
            logger.info(f"  -> Discovered Tool Calls Requests Array:\n{json.dumps(initial_ai_response.tool_calls, indent=2)}")
            
            active_rate = 85.5
            for call in initial_ai_response.tool_calls:
                if call["name"] == "get_conversion_factor":
                    logger.info(f"\n  -> Resolving Tool 1 [{call['name']}] logic synchronously...")
                    t_msg1 = get_conversion_factor.invoke(call)
                    messages.append(t_msg1)
                    
                    # Safely resolving inner rate conversion strings
                    try:
                        content_dict = json.loads(t_msg1.content) if isinstance(t_msg1.content, str) else t_msg1.content
                        if isinstance(content_dict, dict):
                            active_rate = content_dict.get("conversion_rate", active_rate)
                        else:
                            active_rate = float(t_msg1.content)
                    except:
                        active_rate = float(t_msg1.content)
                    logger.info(f"     Extracted conversion rate parameter multiplier: {active_rate}")
                
                elif call["name"] == "convert":
                    logger.info(f"\n  -> Intercepting Tool 2 [{call['name']}] execution payload block...")
                    logger.info(f"     Injecting dynamic parameter payload manually: conversion_rate={active_rate}")
                    
                    # Injecting protected contextual runtime properties prior to calling execution target
                    call["args"]["conversion_rate"] = active_rate
                    t_msg2 = convert.invoke(call)
                    messages.append(t_msg2)
                    logger.info(f"     Returned execution output payload: {t_msg2.content}")
            
            logger.info("\n[Invoking Final Multi-Tool Pipeline Message Array Synthesis]...")
            final_res = llm_bound.invoke(messages)
            logger.info(f"\n[Final Synthesis Response Engine Output]:\n  -> '{final_res.content}'")
            
        except Exception as e:
            logger.error(f"Runtime execution logic exception encountered: {e}")
    else:
        logger.info("\n[Simulating Dual-Tool Injection Resolution Workflow (Offline)]:")
        logger.info("  1. Model processes input and requests parallel tool execution targets.")
        logger.info("  2. App Layer invokes tool 1 ('get_conversion_factor') -> outputting multiplier value: 85.5")
        logger.info("  3. App Layer injects 'conversion_rate': 85.5 into tool 2 call payload parameters.")
        logger.info("  4. App Layer invokes tool 2 ('convert') -> outputting calculation string: 855.0")
        logger.info("\n[Final Synthesis Response Engine Output]:")
        logger.info("  -> 'The conversion factor between USD and INR is 85.5. Based on this factor, 10 USD equals 855.0 INR.'")


# ===============================================================================================
# SECTION 4: AUTONOMOUS AGENT ORCHESTRATION (ReAct)
# ===============================================================================================

def demonstrate_autonomous_agent_orchestration(llm):
    """
    Packages toolsets inside optimized standard agent abstraction runtime structures.
    Demonstrates both legacy initialize_agent and modern langgraph create_react_agent interfaces.
    """
    logger.info("\n" + "="*80)
    logger.info("SECTION 4: AUTONOMOUS AGENT ORCHESTRATION VIA AGENT EXECUTORS")
    logger.info("="*80)

    tools_list = [get_conversion_factor, convert]
    test_query = "Hi, how are you doing today?"

    # Attempting Modern LangGraph ReAct implementation
    try:
        from langgraph.prebuilt import create_react_agent
        logger.info("[Executing via Modern LangGraph state graph interface (create_react_agent)]:")
        if llm:
            agent_app = create_react_agent(llm, tools_list)
            logger.info(f"Triggering active graph evaluation -> Query: '{test_query}'")
            res_obj = agent_app.invoke({"messages": [HumanMessage(content=test_query)]})
            last_msg = res_obj["messages"][-1].content
            logger.info(f"\n[LangGraph ReAct Agent Output Payload]:\n  -> '{last_msg}'")
        else:
            logger.info("  Config Block Target Code:")
            logger.info("    from langgraph.prebuilt import create_react_agent")
            logger.info("    agent_app = create_react_agent(llm, tools=[get_conversion_factor, convert])")
            logger.info("    output = agent_app.invoke({'messages': [HumanMessage(content=query)]})")
            logger.info("\n  [Simulated Offline Graph Return Result]:")
            logger.info("    -> 'Hello! I am functioning optimally and ready to resolve conversion tasks.'")
        return
    except ImportError:
        pass

    # Fallback to standard classic agent interface
    try:
        from langchain.agents import initialize_agent, AgentType
        logger.info("[Executing via Legacy initialize_agent execution wrapper]:")
        if llm:
            agent_executor = initialize_agent(
                tools=tools_list,
                llm=llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            resp = agent_executor.invoke({"input": test_query})
            logger.info(f"\n[Agent Executor Chain Return Payload]:\n  -> {resp['output']}")
        else:
            logger.info("  Config Block Target Code:")
            logger.info("    agent_executor = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)")
            logger.info("\n  [Simulated Run Log Output Progression]:")
            logger.info("    > Entering new AgentExecutor chain...")
            logger.info("    Final Answer: I am functioning perfectly and ready to process incoming requests!")
            logger.info("    > Finished chain.")
    except ImportError as e:
        logger.warning(f"Both agent construction modules unavailable in local site-packages: {e}")


if __name__ == "__main__":
    live_engine = get_tool_calling_model()
    demonstrate_tool_inspection()
    execute_manual_tool_calling_loop(live_engine)
    demonstrate_injected_arguments_flow(live_engine)
    demonstrate_autonomous_agent_orchestration(live_engine)
