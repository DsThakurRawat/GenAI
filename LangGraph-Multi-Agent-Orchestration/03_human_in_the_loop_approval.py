import operator
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ============================================================================
# 03. HUMAN-IN-THE-LOOP (HITL) APPROVAL INTERRUPTS
# ============================================================================
# This script demonstrates enterprise security architectures intercepting agent
# loops dynamically before triggering high-risk tool operations. The workflow
# pauses execution states synchronously to await UI continuation parameters.

# --- 1. Define Execution State Schema ---
class InterceptState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    action_requested: str
    is_approved: bool

# --- 2. Define Execution Nodes ---
def node_agent_planner(state: InterceptState) -> dict:
    """Simulates an AI agent requesting authorization to run critical tools."""
    print("\n--- [AGENT PLANNER] Formulating sensitive transactional command requests ---")
    req_action = "DROP TABLE production_records;"
    print(f"-> Threat detection alert: Planning execution of target string: '{req_action}'")
    return {
        "messages": [AIMessage(content=f"Requesting authorization to execute command: {req_action}")],
        "action_requested": req_action,
        "is_approved": False
    }

def node_sensitive_tool(state: InterceptState) -> dict:
    """Executes target transactional logic strictly upon verifying approval bounds."""
    print("\n--- [SENSITIVE TOOL ENGINE] Verifying runtime security policy permissions ---")
    if state.get("is_approved", False):
        exec_status = "[EXECUTED]: Low-level production tables mutated successfully."
    else:
        exec_status = "[REJECTED]: Execution dropped. Unverified permission flags detected."
        
    print(f"-> Operational validation output: {exec_status}")
    return {"messages": [AIMessage(content=exec_status)]}

# --- 3. Assemble Graph with Static Checkpoint Interceptions ---
builder = StateGraph(InterceptState)

builder.add_node("node_agent_planner", node_agent_planner)
builder.add_node("node_sensitive_tool", node_sensitive_tool)

builder.add_edge(START, "node_agent_planner")
builder.add_edge("node_agent_planner", "node_sensitive_tool")
builder.add_edge("node_sensitive_tool", END)

# Compile using native interrupt parameters to trap execution buffers
memory_tracker = MemorySaver()
safe_app = builder.compile(
    checkpointer=memory_tracker,
    interrupt_before=["node_sensitive_tool"]  # Pause completely prior to tool invocation
)

# --- 4. Simulated Interactive UI Execution Block ---
if __name__ == "__main__":
    print("🚀 Running Secure HITL Runtime Graph Pipeline...")
    
    session_thread = {"configurable": {"thread_id": "secure_session_999"}}
    initial_payload = {
        "messages": [HumanMessage(content="Clear stale testing metrics.")],
        "action_requested": "",
        "is_approved": False
    }
    
    # 1. Execute up until interrupt breakpoint node is hit
    print("\n[Phase 1]: Evaluating base application state tasks...")
    for event in safe_app.stream(initial_payload, config=session_thread):
        pass
        
    # Verify graph state shows paused execution checkpoint
    active_snapshot = safe_app.get_state(session_thread)
    print(f"\n⏸️ Checkpoint Trap Reached! Active Interrupted Step Target: {active_snapshot.next}")
    
    # 2. Simulate User Admin checking logs and granting explicit authorization updates
    print("\n[Phase 2]: Receiving secure UI continuation input authorization token...")
    # Inject state mutations directly into paused Checkpoint memory cache
    safe_app.update_state(
        config=session_thread,
        values={"is_approved": True},
        as_node="node_agent_planner"
    )
    
    # 3. Resume graph processing from breakpoint target node
    print("\n[Phase 3]: Resuming interrupted operational task nodes...")
    for event in safe_app.stream(None, config=session_thread):
        for node_id, payload in event.items():
            print(f"-> Synchronized block output summary from: '{node_id}'")
            
    print("\n✅ Human-in-the-Loop Interception Scenario Validated Successfully.")
