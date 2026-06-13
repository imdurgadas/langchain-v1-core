# create a file: 06_langgraph_memory.py
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def get_server_status(server_name: str) -> str:
    """Returns the current operational status of a named server."""
    servers = {
        "api-prod-01": "HEALTHY — 99.98% uptime.",
        "db-primary":  "WARNING — High CPU at 87%.",
    }
    return servers.get(server_name.lower(), f"Server '{server_name}' not found.")

graph_tools = [get_server_status]
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0).bind_tools(graph_tools)

def run_agent(state: GraphState):
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(GraphState)
builder.add_node("agent", run_agent)
builder.add_node("tools", ToolNode(graph_tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {True: "tools", False: END})
builder.add_edge("tools", "agent")

# MemorySaver stores state in memory (process lifetime only)
# For production, swap this with SqliteSaver or RedisSaver
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# thread_id isolates conversations — different users get different thread IDs
config = {"configurable": {"thread_id": "ops-session-001"}}

# --- Turn 1 ---
print("=== Turn 1 ===")
r1 = graph.invoke({"messages": [HumanMessage("Check the status of db-primary")]}, config)
print(r1["messages"][-1].content)

# --- Turn 2 (no tool call needed — model uses its memory of Turn 1) ---
print("\n=== Turn 2 ===")
r2 = graph.invoke({"messages": [HumanMessage("Should I be worried about what you found?")]}, config)
print(r2["messages"][-1].content)
