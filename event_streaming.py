# create a file: 26_event_streaming.py
import asyncio
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# -------------------------------------------------------
# A simple single-chain example to understand event types
# before adding the complexity of a full agent graph
# -------------------------------------------------------
headline_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a news headline writer for Tech News Daily."),
        ("human", "Write 3 headline options for this story: {story}")
    ])
    | llm.with_config({"run_name": "headline_llm"})  # name shows up in events
    | StrOutputParser()
)

async def inspect_events():
    """Print every event with its type and relevant data."""
    
    story = "Cloudflare reported a 99.99% uptime SLA breach affecting European customers for 47 minutes on June 13, 2026. The cause was a BGP routing misconfiguration."
    
    print("=== Raw Event Stream ===\n")
    
    async for event in headline_chain.astream_events(
        {"story": story},
        version="v2"          # always specify version — v2 is current stable
    ):
        kind = event["event"]
        name = event.get("name", "unknown")
        
        # Skip verbose start/end events for this inspection
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:  # filter empty chunks
                print(f"  TOKEN: '{chunk}'", end="", flush=True)
        
        elif kind == "on_tool_start":
            print(f"\n  TOOL START → {event['data']['input']}")
        
        elif kind == "on_tool_end":
            print(f"  TOOL END  → output length: {len(str(event['data']['output']))}")
        
        elif kind in ("on_chain_start", "on_chain_end"):
            print(f"\n  {kind.upper()}: {name}")
        
        elif kind == "on_chat_model_end":
            print(f"\n\n  MODEL DONE — total tokens used: {event['data']['output'].usage_metadata}")


asyncio.run(inspect_events())
