# create a file: 02_messages.py
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# This is a manually constructed conversation history
# In a real agent, this list is built automatically as the agent runs
conversation = [
    SystemMessage(content="You are a concise data analyst. Respond only with bullet points."),
    HumanMessage(content="What is the difference between data replication and sharding?"),
]

# Load your model and invoke with the full history
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()

model = init_chat_model("gemini-3.5-flash", model_provider="google-genai")
response = model.invoke(conversation)
print(response.content)
