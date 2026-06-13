# create a file: 16_supervisor.py
import json
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Use a capable model for the supervisor — it needs to reason about task state
supervisor_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

AGENTS = ["researcher", "writer", "notifier"]

def supervisor_node(state: dict) -> dict:
    """
    The supervisor reads the current state and decides which specialist
    to invoke next, or returns FINISH when the task is complete.
    
    The supervisor does NOT execute tasks. It only orchestrates.
    This separation is critical — mixing orchestration and execution
    in one agent creates unpredictable, hard-to-debug behaviour.
    """
    task = state["task"]
    research_done = bool(state.get("research_findings", "").strip())
    writing_done = bool(state.get("draft_content", "").strip())
    
    # Check message history to see if notifier has already run
    notifier_ran = any(
        getattr(m, "name", "") == "notifier" 
        for m in state.get("messages", [])
    )
    
    system_prompt = f"""You are a supervisor managing a team of specialist agents.
You must decide which agent to invoke next to complete the task.

Available agents:
- researcher: Gathers and synthesises factual information. Use FIRST.
- writer: Transforms research findings into polished prose. Use AFTER researcher.  
- notifier: Formats and delivers the final output. Use AFTER writer. Use LAST.
- FINISH: The task is fully complete. Use when notifier has run.

Current task: {task}

Current state:
- Research completed: {research_done}
- Writing completed: {writing_done}  
- Notification sent: {notifier_ran}

Respond with ONLY a JSON object like this: {{"next": "researcher"}}
Choose one of: researcher, writer, notifier, FINISH"""

    response = supervisor_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Which agent should act next?")
    ])
    
    # Parse the routing decision
    try:
        # Handle cases where the model wraps JSON in markdown code fences
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        decision = json.loads(content.strip())
        next_agent = decision.get("next", "FINISH")
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, use deterministic fallback logic
        if not research_done:
            next_agent = "researcher"
        elif not writing_done:
            next_agent = "writer"
        elif not notifier_ran:
            next_agent = "notifier"
        else:
            next_agent = "FINISH"
    
    # Validate the decision is one of the known agents
    valid_options = AGENTS + ["FINISH"]
    if next_agent not in valid_options:
        next_agent = "FINISH"
    
    print(f"\n[Supervisor] Routing to: {next_agent}")
    return {"next_agent": next_agent}


# Test the supervisor in isolation
if __name__ == "__main__":
    # Simulate state at different stages
    stages = [
        {"task": "Research vector databases", "research_findings": "", "draft_content": "", "messages": []},
        {"task": "Research vector databases", "research_findings": "1. Vector DBs store embeddings...", "draft_content": "", "messages": []},
        {"task": "Research vector databases", "research_findings": "1. Vector DBs store embeddings...", "draft_content": "Executive summary...", "messages": []},
    ]
    
    for i, state in enumerate(stages):
        print(f"\n=== Stage {i+1} ===")
        result = supervisor_node(state)
        print(f"Decision: {result['next_agent']}")
    
    print("\n✓ Supervisor routing logic tested.")
