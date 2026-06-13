# create a file: 18_multi_agent_memory.py
import os
import json
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

class ResearchTeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    research_findings: str
    draft_content: str
    next_agent: str


def researcher_agent(state: ResearchTeamState) -> dict:
    task = state["task"]
    # If we already have findings, allow revision based on latest message
    existing = state.get("research_findings", "")
    context = f"\nExisting findings (revise if needed):\n{existing}" if existing else ""
    
    response = llm.invoke([HumanMessage(content=f"""Research specialist. Task: {task}{context}
Produce structured numbered findings. Be specific.""")])
    print(f"  [Researcher] Updated findings.")
    return {
        "research_findings": response.content,
        "messages": [AIMessage(content=response.content, name="researcher")]
    }

def writer_agent(state: ResearchTeamState) -> dict:
    # Check if there's a revision instruction in the latest message
    latest_messages = state.get("messages", [])
    revision_note = ""
    for m in reversed(latest_messages):
        if hasattr(m, "type") and m.type == "human":
            revision_note = f"\nRevision instruction from user: {m.content}"
            break
    
    response = llm.invoke([HumanMessage(content=f"""Writing specialist.
Task: {state['task']}
Research: {state['research_findings']}{revision_note}
Write a 3-paragraph executive summary.""")])
    print(f"  [Writer] Draft updated.")
    return {
        "draft_content": response.content,
        "messages": [AIMessage(content=response.content, name="writer")]
    }

def notifier_agent(state: ResearchTeamState) -> dict:
    print(f"  [Notifier] Final message ready.")
    return {
        "messages": [AIMessage(
            content=f"Delivered: {state['draft_content'][:100]}...",
            name="notifier"
        )]
    }

def supervisor_node(state: ResearchTeamState) -> dict:
    research_done = bool(state.get("research_findings", "").strip())
    writing_done = bool(state.get("draft_content", "").strip())
    notifier_ran = any(getattr(m, "name", "") == "notifier" for m in state.get("messages", []))

    if not research_done:
        next_agent = "researcher"
    elif not writing_done:
        next_agent = "writer"
    elif not notifier_ran:
        next_agent = "notifier"
    else:
        next_agent = "FINISH"
    
    print(f"\n[Supervisor] → {next_agent}")
    return {"next_agent": next_agent}

def route_to_agent(state: ResearchTeamState) -> str:
    return END if state.get("next_agent") == "FINISH" else state.get("next_agent", END)

# Build graph with MemorySaver checkpointer
def build_persistent_graph():
    graph = StateGraph(ResearchTeamState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("notifier", notifier_agent)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_to_agent,
        {"researcher": "researcher", "writer": "writer", "notifier": "notifier", END: END})
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("notifier", "supervisor")
    
    # MemorySaver persists state between invocations on the same thread_id
    return graph.compile(checkpointer=MemorySaver())

graph = build_persistent_graph()

# Session 1: First run
print("=" * 60)
print("Session 1: Initial research task")
config = {"configurable": {"thread_id": "project-alpha-001"}}

result = graph.invoke({
    "task": "Summarise the advantages of LangGraph over ReAct agents",
    "messages": [],
    "research_findings": "",
    "draft_content": "",
    "next_agent": "",
}, config, {"recursion_limit": 20})

print(f"\n✓ Session 1 complete.")
print(f"Draft preview: {result['draft_content'][:200]}...")

# Session 2: Resume and request revision
# This picks up from where session 1 left off — the existing findings and draft are in state
print("\n" + "=" * 60)
print("Session 2: Requesting revision (same thread_id)")

# Force a re-draft by clearing draft_content but keeping research
result2 = graph.invoke({
    "task": "Summarise the advantages of LangGraph over ReAct agents",
    "messages": [HumanMessage(content="Please revise the summary to focus specifically on production use cases and real-world benefits.")],
    "research_findings": result["research_findings"],  # preserve existing research
    "draft_content": "",  # clear draft to trigger re-writing
    "next_agent": "",
}, config, {"recursion_limit": 20})

print(f"\n✓ Session 2 complete. Revised draft:")
print(result2['draft_content'])
