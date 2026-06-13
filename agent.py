# create a file: 04_agent.py
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

# Step 1: Define a tool — a normal Python function decorated with @tool
# The docstring is critical: the model reads it to decide WHEN to call this tool
@tool
def check_stock_inventory(sku_id: str) -> str:
    """
    Retrieves current stock levels and warehouse locations for a given SKU ID.
    Use this tool when the user asks about product availability, stock counts, or fulfillment.
    """
    sku = sku_id.strip().upper()
    if "SKU-99" in sku:
        return "Warehouse East: 42 units. Warehouse West: Out of stock."
    elif "SKU-10" in sku:
        return "Warehouse Central: 500 units. Overnight shipping available."
    else:
        return "SKU not found in catalog. Request forwarded to distribution team."

# Step 2: Load the standard ReAct prompt from LangChain Hub
# This prompt teaches the model HOW to reason, act, and observe in a loop
react_prompt = hub.pull("hwchase17/react")

# Step 3: Load the model and assemble the agent
model = init_chat_model("gemini-3.5-flash", model_provider="google-genai")
agent_tools = [check_stock_inventory]

agent = create_react_agent(model, agent_tools, react_prompt)

# Step 4: Wrap in an executor — this actually runs the ReAct loop
executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True)

# Step 5: Run it — verbose=True prints every Thought/Action/Observation step
result = executor.invoke({"input": "Can we fulfil an order for 20 units of SKU-99 right now?"})
print("\n--- Final Answer ---")
print(result["output"])
