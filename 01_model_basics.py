# create a file: 01_model_basics.py
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

# Both models are initialised the same way — just change the string
gemini_flash = init_chat_model("gemini-3.5-flash", model_provider="google-genai")
gemini_pro   = init_chat_model("gemini-3.5-pro",   model_provider="google-genai")

response = gemini_flash.invoke([HumanMessage(content="Explain what an AI agent is in two sentences.")])
print(response.content)
