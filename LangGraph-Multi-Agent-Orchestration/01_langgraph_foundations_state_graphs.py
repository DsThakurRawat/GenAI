import os
import operator
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ============================================================================
# 01. LANGGRAPH FOUNDATIONS: STATE MACHINE ORCHESTRATION & CHECKPOINTERS
# ============================================================================
# This script demonstrates the core architectural pattern of modern Agentic AI:
# Defining custom State dictionaries, wiring execution Nodes, defining dynamic
# conditional Edges, and maintaining complete time-travel checkpointers.

# --- 1. Define the Graph State Schema ---
# We use operator.add to instruct the StateGraph reducer to append new messages
# to the existing list, rather than overwriting the entire list attribute.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    iteration_count: int

# --- 2. Initialize Core Models ---
# Ensure your OPENAI_API_KEY environment variable is exported correctly.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# --- 3. Define Graph Node Processing Functions ---
def node_agent(state: AgentState) -> dict:
    """Core evaluation node passing state messages to the inference layer."""
    print("\n--- [NODE: AGENT] Evaluating active state context ---")
    current_messages = state.get("messages", [])
    current_count = state.get("iteration_count", 0)
    
    # Evaluate via LLM
    response = llm.invoke(current_messages)
    
    # Return dictionary delta to be merged into global state by the reducer
    return {
        "messages": [response],
        "iteration_count": current_count + 1
    }

def node_reflector(state: AgentState) -> dict:
    """Auxiliary node that adds structural reflection metrics to the dialogue."""
    print("--- [NODE: REFLECTOR] Injecting cognitive reflection logs ---")
    reflection_msg = AIMessage(content="[Reflection Check]: Verified factual bounds cleanly.")
    return {"messages": [reflection_msg]}

# --- 4. Define Dynamic Conditional Edge Routing Logic ---
def route_next_step(state: AgentState) -> Literal["node_reflector", "__end__"]:
    """Determines subsequent graph execution targets based on state limits."""
    count = state.get("iteration_count", 0)
    print(f"--- [ROUTER] Evaluating cycle threshold limit (Current: {count}) ---")
    
    # Loop back to reflection if iteration bounds are under target thresholds
    if count < 2:
        return "node_reflector"
    return "__end__"

# --- 5. Compile the Final State Graph ---
workflow = StateGraph(AgentState)

# Map target nodes
workflow.add_node("node_agent", node_agent)
workflow.add_node("node_reflector", node_reflector)

# Map edge connections
workflow.add_edge(START, "node_agent")
workflow.add_conditional_edges("node_agent", route_next_step)
workflow.add_edge("node_reflector", "node_agent")

# Instantiate thread checkpoint memory persistence
memory_checkpointer = MemorySaver()
app = workflow.compile(checkpointer=memory_checkpointer)

# --- 6. Execution Verification Block ---
if __name__ == "__main__":
    print("🚀 Booting persistent LangGraph runtime State Machine...")
    
    # Define an explicit thread mapping to track unique user sessions
    thread_config = {"configurable": {"thread_id": "session_alpha_001"}}
    
    initial_input = {
        "messages": [HumanMessage(content="Explain the structural difference between DAGs and State Machines.")],
        "iteration_count": 0
    }
    
    # Execute graph logic stream
    for output_event in app.stream(initial_input, config=thread_config):
        for node_id, node_output in output_event.items():
            print(f"\nCompleted execution payload from Node: '{node_id}'")
            # Inspect incremental changes
            latest_msg = node_output.get("messages", [])[-1]
            print(f"-> Snapshot preview: {latest_msg.content[:120]}...")

    print("\n✅ Stateful Multi-Node Execution Sequence Completed Successfully.")
