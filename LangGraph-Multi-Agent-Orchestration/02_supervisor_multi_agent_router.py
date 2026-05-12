import operator
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ============================================================================
# 02. SUPERVISOR MULTI-AGENT ROUTING ARCHITECTURE
# ============================================================================
# This script demonstrates coordinating multiple decoupled expert sub-agents
# utilizing a main "Supervisor" router agent. The supervisor dynamically maps
# execution paths between specialized functional sub-systems.

# --- 1. Define Collaborative Graph State Schema ---
class CollaborationState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next_target: str

# --- 2. Initialize Infrastructure ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# --- 3. Define Specialized Sub-Agent Nodes ---
def node_coder(state: CollaborationState) -> dict:
    """Specialist sub-agent focused strictly on code synthesis tasks."""
    print("\n--- [SPECIALIST AGENT: CODER] Evaluating execution logic stream ---")
    sys_prompt = SystemMessage(content="You are an expert Python coder. Output clean code snippets.")
    response = llm.invoke([sys_prompt] + state["messages"])
    return {"messages": [response]}

def node_researcher(state: CollaborationState) -> dict:
    """Specialist sub-agent focused strictly on factual background lookups."""
    print("\n--- [SPECIALIST AGENT: RESEARCHER] Performing context review ---")
    sys_prompt = SystemMessage(content="You are an expert AI Researcher. Provide precise theoretical summary context.")
    response = llm.invoke([sys_prompt] + state["messages"])
    return {"messages": [response]}

# --- 4. Define Centralized Supervisor Router Engine ---
def node_supervisor(state: CollaborationState) -> dict:
    """Evaluates session state context to coordinate target sub-agent routing."""
    print("\n👑 --- [SUPERVISOR ROUTER] Inspecting active collaborative thread ---")
    last_msg = state["messages"][-1].content.lower()
    
    # Analyze text requests to infer intended downstream recipient destinations
    if "code" in last_msg or "script" in last_msg:
        decision = "node_coder"
    elif "research" in last_msg or "paper" in last_msg or "theory" in last_msg:
        decision = "node_researcher"
    else:
        decision = "FINISH"
        
    print(f"-> Routing evaluation conclusion selected path: '{decision}'")
    return {"next_target": decision}

# --- 5. Implement Dynamic Conditional Graph Sweeper ---
def conditional_router(state: CollaborationState) -> Literal["node_coder", "node_researcher", "__end__"]:
    """Resolves dynamic graph node jumps explicitly."""
    target = state.get("next_target", "FINISH")
    if target == "node_coder":
        return "node_coder"
    elif target == "node_researcher":
        return "node_researcher"
    return "__end__"

# --- 6. Assemble Multi-Agent DAG Topology ---
builder = StateGraph(CollaborationState)

# Wire nodes
builder.add_node("node_supervisor", node_supervisor)
builder.add_node("node_coder", node_coder)
builder.add_node("node_researcher", node_researcher)

# Connect cyclic loop routes
builder.add_edge(START, "node_supervisor")
builder.add_conditional_edges("node_supervisor", conditional_router)
builder.add_edge("node_coder", "node_supervisor")
builder.add_edge("node_researcher", "node_supervisor")

multi_agent_system = builder.compile()

# --- 7. Operational Validation Block ---
if __name__ == "__main__":
    print("🚀 Initiating Collaborative Multi-Agent System Interface...")
    
    task_payload = {
        "messages": [HumanMessage(content="Write a Python script demonstrating closures.")],
        "next_target": ""
    }
    
    # Execute graph iteration engine
    final_output_state = multi_agent_system.invoke(task_payload)
    
    print("\n🏁 Collaborative Multi-Agent Output Buffer Result:")
    for msg in final_output_state["messages"]:
        print(f"[{msg.__class__.__name__}]: {msg.content[:150]}...")
