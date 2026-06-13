# create a file: 14_multi_agent_state.py
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ResearchTeamState(TypedDict):
    """
    Shared state for the entire multi-agent research pipeline.
    
    Each field is a channel — agents read from it and write updates.
    The 'messages' field uses the add_messages reducer so that each
    agent's output is appended rather than overwriting the history.
    """
    # Full conversation history across all agents
    messages: Annotated[list[BaseMessage], add_messages]
    
    # The original task from the user
    task: str
    
    # Populated by the Researcher Agent
    research_findings: str
    
    # Populated by the Writer Agent  
    draft_content: str
    
    # The supervisor's routing decision (which agent to call next)
    next_agent: str

print("State schema defined successfully.")
print("Fields:", list(ResearchTeamState.__annotations__.keys()))
