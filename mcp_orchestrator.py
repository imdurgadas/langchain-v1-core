# Save as: 10_mcp_orchestrator.py
import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

async def run():
    async with MultiServerMCPClient() as client:

        # Connect to the stdio server — the client launches mcp_server_data.py
        # as a subprocess automatically
        await client.connect_server(
            "data_utils",
            command="python",
            args=["mcp_server_data.py"],
            transport="stdio"
        )

        # Connect to the already-running HTTP server
        await client.connect_server(
            "network_utils",
            url="http://127.0.0.1:8765/mcp",
            transport="http"
        )

        # get_tools() discovers all tools from all connected servers
        # The agent sees them as a flat list — it has no idea which server each comes from
        tools = await client.get_tools()
        print(f"Discovered {len(tools)} tools:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        agent = create_react_agent(llm, tools)

        # The agent will automatically pick the right tool for each part of this query
        result = await agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": "Convert 37 Celsius to Fahrenheit, then check if production.api.company.com is reachable, and tell me the latency profile for us-east-1."
            }]
        })

        print("\n--- Final Answer ---")
        print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(run())
