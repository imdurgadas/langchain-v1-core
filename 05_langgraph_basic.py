# create a file: 05_langgraph_basic.py
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# -------------------------------------------------------
# 1. Define the State
# -------------------------------------------------------
# add_messages is a "reducer" — it APPENDS new messages to the list
# rather than replacing the whole list. This is what preserves history.
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# -------------------------------------------------------
# 2. Define a Tool
# -------------------------------------------------------
@tool
def get_server_status(server_name: str) -> str:
    """
    Returns the current operational status of a named server.
    Use this when the user asks about server health, uptime, or status.
    """
    servers = {
        "api-prod-01": "HEALTHY — 99.98% uptime. Last incident: 14 days ago.",
        "db-primary":  "WARNING — High CPU at 87%. Recommend investigation.",
        "cache-01":    "HEALTHY — 100% uptime. Memory usage nominal.",
    }
    return servers.get(server_name.lower(), f"Server '{server_name}' not found in registry.")

graph_tools = [get_server_status]

# -------------------------------------------------------
# 3. Set up the LLM with tools bound to it
# -------------------------------------------------------
# bind_tools() attaches the tool schemas to the model so it knows
# what tools exist and when to request them
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0).bind_tools(graph_tools)

# -------------------------------------------------------
# 4. Define the Node function
# -------------------------------------------------------
def run_agent(state: GraphState):
    response = llm.invoke(state["messages"])
    # Return a dict — LangGraph merges this into the state via the reducer
    return {"messages": [response]}

# -------------------------------------------------------
# 5. Build the graph
# -------------------------------------------------------
builder = StateGraph(GraphState)

builder.add_node("agent", run_agent)
builder.add_node("tools", ToolNode(graph_tools))

builder.add_edge(START, "agent")

# tools_condition checks if the last AIMessage has tool_calls
# If yes -> go to "tools". If no -> go to END.
builder.add_conditional_edges("agent", tools_condition, {True: "tools", False: END})

# After tools run, always go back to agent for the next reasoning step
builder.add_edge("tools", "agent")

graph = builder.compile()

# -------------------------------------------------------
# 6. Run it
# -------------------------------------------------------
result = graph.invoke({"messages": [HumanMessage(content="What's the status of db-primary?")]})
print(result["messages"][-1].content)
