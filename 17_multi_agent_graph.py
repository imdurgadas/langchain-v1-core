# create a file: 17_multi_agent_graph.py
import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import json

load_dotenv()

# -------------------------------------------------------
# State definition (consolidated here for a self-contained file)
# -------------------------------------------------------
class ResearchTeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    research_findings: str
    draft_content: str
    next_agent: str


# -------------------------------------------------------
# Specialist agents
# -------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

def researcher_agent(state: ResearchTeamState) -> dict:
    task = state["task"]
    response = llm.invoke([HumanMessage(content=f"""You are a specialist research agent.
Task: {task}
Produce a structured list of key findings as numbered bullet points. Be specific and factual.""")])
    findings = response.content
    print(f"  [Researcher] Done. {len(findings)} chars of findings.")
    return {
        "research_findings": findings,
        "messages": [AIMessage(content=findings, name="researcher")]
    }

def writer_agent(state: ResearchTeamState) -> dict:
    findings = state["research_findings"]
    task = state["task"]
    response = llm.invoke([HumanMessage(content=f"""You are a specialist writing agent.
Original task: {task}
Research findings:
{findings}
Write a 3-paragraph executive summary based strictly on these findings.""")])
    draft = response.content
    print(f"  [Writer] Done. {len(draft)} chars of draft.")
    return {
        "draft_content": draft,
        "messages": [AIMessage(content=draft, name="writer")]
    }

def notifier_agent(state: ResearchTeamState) -> dict:
    draft = state["draft_content"]
    task = state["task"]
    slack_message = f"*📋 Research Summary*\n_Task: {task}_\n\n{draft}"
    print(f"  [Notifier] Message ready ({len(slack_message)} chars). Would send to Slack.")
    return {
        "messages": [AIMessage(
            content=f"Delivery complete. Message: {slack_message[:200]}...",
            name="notifier"
        )]
    }


# -------------------------------------------------------
# Supervisor
# -------------------------------------------------------
supervisor_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

def supervisor_node(state: ResearchTeamState) -> dict:
    research_done = bool(state.get("research_findings", "").strip())
    writing_done = bool(state.get("draft_content", "").strip())
    notifier_ran = any(getattr(m, "name", "") == "notifier" for m in state.get("messages", []))

    system_prompt = f"""You are a supervisor managing specialist agents.
Task: {state['task']}
State: research_done={research_done}, writing_done={writing_done}, notifier_ran={notifier_ran}
Agents: researcher (first), writer (after research), notifier (after writing), FINISH (when notifier ran).
Respond ONLY with JSON: {{"next": "agent_name"}}"""

    response = supervisor_llm.invoke([HumanMessage(content=system_prompt)])
    
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        decision = json.loads(content)
        next_agent = decision.get("next", "FINISH")
    except Exception:
        # Deterministic fallback
        if not research_done:
            next_agent = "researcher"
        elif not writing_done:
            next_agent = "writer"
        elif not notifier_ran:
            next_agent = "notifier"
        else:
            next_agent = "FINISH"
    
    valid = ["researcher", "writer", "notifier", "FINISH"]
    next_agent = next_agent if next_agent in valid else "FINISH"
    
    print(f"\n[Supervisor] → {next_agent}")
    return {"next_agent": next_agent}


# -------------------------------------------------------
# Routing function
# Called after the supervisor node runs — reads next_agent and
# returns the string name of the next node to invoke.
# -------------------------------------------------------
def route_to_agent(state: ResearchTeamState) -> str:
    """
    This function is the conditional edge out of the supervisor node.
    LangGraph calls it with the current state and uses the return value
    to decide which node to activate next.
    """
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent


# -------------------------------------------------------
# Build and compile the graph
# -------------------------------------------------------
def build_research_graph():
    graph = StateGraph(ResearchTeamState)
    
    # Register all nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("notifier", notifier_agent)
    
    # Entry point: always start at the supervisor
    graph.add_edge(START, "supervisor")
    
    # The supervisor uses a conditional edge to route dynamically
    graph.add_conditional_edges(
        "supervisor",          # from this node
        route_to_agent,        # call this function to decide where to go
        {                      # map return values to node names
            "researcher": "researcher",
            "writer": "writer",
            "notifier": "notifier",
            END: END,
        }
    )
    
    # After each specialist completes, return control to the supervisor
    # This allows the supervisor to re-evaluate state and decide what's next
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("notifier", "supervisor")
    
    return graph.compile()


# -------------------------------------------------------
# Run the graph
# -------------------------------------------------------
if __name__ == "__main__":
    research_graph = build_research_graph()
    
    initial_state = {
        "task": "Summarise the key advantages of LangGraph over a simple ReAct agent loop for building production AI systems.",
        "messages": [],
        "research_findings": "",
        "draft_content": "",
        "next_agent": "",
    }
    
    print("=" * 60)
    print("Starting Multi-Agent Research Pipeline")
    print(f"Task: {initial_state['task']}")
    print("=" * 60)
    
    # Stream execution so we see each step as it happens
    for step in research_graph.stream(initial_state, {"recursion_limit": 20}):
        node_name = list(step.keys())[0]
        print(f"\n--- Step completed: {node_name} ---")
    
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    
    # Get the final state
    final = research_graph.invoke(initial_state, {"recursion_limit": 20})
    print("\n=== Final Draft ===")
    print(final["draft_content"])
