# create a file: 27_live_dashboard.py
import asyncio
import os
import sys
import time
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0, streaming=True)

# -------------------------------------------------------
# Newsroom state (same shape as Part 6)
# -------------------------------------------------------
class NewsroomState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    research_findings: str
    draft_content: str
    next_agent: str


# -------------------------------------------------------
# Dashboard renderer — prints status without clearing the screen
# Compatible with any terminal (no ANSI tricks needed)
# -------------------------------------------------------
class NewsDashboard:
    def __init__(self):
        self.current_agent = None
        self.token_count = 0
        self.start_time = time.time()
        self.tool_calls = []
    
    def agent_started(self, agent_name: str):
        elapsed = time.time() - self.start_time
        print(f"\n[{elapsed:.1f}s] ▶ {agent_name.upper()} started", flush=True)
        self.current_agent = agent_name
    
    def agent_token(self, token: str):
        print(token, end="", flush=True)
        self.token_count += 1
    
    def agent_done(self, agent_name: str):
        elapsed = time.time() - self.start_time
        print(f"\n[{elapsed:.1f}s] ✓ {agent_name.upper()} done", flush=True)
    
    def tool_called(self, tool_name: str, input_summary: str):
        elapsed = time.time() - self.start_time
        print(f"\n[{elapsed:.1f}s] 🔧 Tool: {tool_name} | Input: {input_summary[:60]}", flush=True)
        self.tool_calls.append(tool_name)
    
    def summary(self):
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Pipeline complete in {elapsed:.1f}s")
        print(f"Total tokens streamed: {self.token_count}")
        print(f"Tool calls: {self.tool_calls or 'none'}")


dashboard = NewsDashboard()


# -------------------------------------------------------
# Specialist agents — each named for event filtering
# -------------------------------------------------------
def make_researcher(llm):
    from langchain_core.prompts import ChatPromptTemplate
    return (
        ChatPromptTemplate.from_messages([
            ("system", "You are the Tech News Daily researcher. Gather key facts."),
            ("human", "Research task: {task}\nProvide 4 numbered findings.")
        ])
        | llm.with_config({"run_name": "researcher_llm"})
    )

def make_writer(llm):
    from langchain_core.prompts import ChatPromptTemplate
    return (
        ChatPromptTemplate.from_messages([
            ("system", "You are the Tech News Daily writer. Write sharp, factual summaries."),
            ("human", "Task: {task}\n\nFindings:\n{findings}\n\nWrite a 2-paragraph summary.")
        ])
        | llm.with_config({"run_name": "writer_llm"})
    )

researcher_chain = make_researcher(llm)
writer_chain = make_writer(llm)


# -------------------------------------------------------
# Graph nodes — each is also named for event filtering
# -------------------------------------------------------
async def researcher_node(state: NewsroomState) -> dict:
    response = await researcher_chain.ainvoke(
        {"task": state["task"]},
        config={"run_name": "researcher_node"}
    )
    return {
        "research_findings": response.content,
        "messages": [AIMessage(content=response.content, name="researcher")]
    }

async def writer_node(state: NewsroomState) -> dict:
    response = await writer_chain.ainvoke(
        {"task": state["task"], "findings": state["research_findings"]},
        config={"run_name": "writer_node"}
    )
    return {
        "draft_content": response.content,
        "messages": [AIMessage(content=response.content, name="writer")]
    }

async def notifier_node(state: NewsroomState) -> dict:
    content = f"📰 PUBLISHED: {state['draft_content'][:200]}..."
    return {
        "messages": [AIMessage(content=content, name="notifier")]
    }

def supervisor_node(state: NewsroomState) -> dict:
    research_done = bool(state.get("research_findings", "").strip())
    writing_done = bool(state.get("draft_content", "").strip())
    notifier_ran = any(getattr(m, "name", "") == "notifier" for m in state.get("messages", []))
    
    if not research_done:
        return {"next_agent": "researcher"}
    elif not writing_done:
        return {"next_agent": "writer"}
    elif not notifier_ran:
        return {"next_agent": "notifier"}
    else:
        return {"next_agent": "FINISH"}

def route_to_agent(state: NewsroomState) -> str:
    n = state.get("next_agent", "FINISH")
    return END if n == "FINISH" else n


# Build the graph
graph = StateGraph(NewsroomState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("notifier", notifier_node)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_to_agent,
    {"researcher": "researcher", "writer": "writer", "notifier": "notifier", END: END})
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
graph.add_edge("notifier", "supervisor")
newsroom_graph = graph.compile()


# -------------------------------------------------------
# The streaming runner — the key logic is here
# -------------------------------------------------------
async def run_with_dashboard(task: str):
    print(f"\n{'='*60}")
    print(f"Tech News Daily — Processing: {task[:70]}")
    print(f"{'='*60}")
    
    initial_state = {
        "task": task,
        "messages": [],
        "research_findings": "",
        "draft_content": "",
        "next_agent": "",
    }
    
    # Track which agent is "active" for event attribution
    active_agent = None
    
    async for event in newsroom_graph.astream_events(
        initial_state,
        version="v2",
        config={"recursion_limit": 20}
    ):
        kind = event["event"]
        name = event.get("name", "")
        
        # Detect which agent node is starting
        if kind == "on_chain_start" and name in ("researcher", "writer", "notifier", "supervisor"):
            dashboard.agent_started(name)
            active_agent = name
        
        # Stream tokens from the LLM — filter to researcher and writer
        elif kind == "on_chat_model_stream":
            if active_agent in ("researcher", "writer"):
                chunk = event["data"]["chunk"].content
                if chunk:
                    dashboard.agent_token(chunk)
        
        # Tool calls (if any are used in future)
        elif kind == "on_tool_start":
            tool_name = name
            tool_input = str(event["data"].get("input", ""))
            dashboard.tool_called(tool_name, tool_input)
        
        # Agent node completed
        elif kind == "on_chain_end" and name in ("researcher", "writer", "notifier"):
            dashboard.agent_done(name)
    
    dashboard.summary()


# -------------------------------------------------------
# Run it
# -------------------------------------------------------
story_task = "Stripe announced native stablecoin payment support for USD Coin (USDC) and Tether (USDT), targeting cross-border B2B payments. The feature launches first in 25 countries with settlement in under 2 minutes."

asyncio.run(run_with_dashboard(story_task))
