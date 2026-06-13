# create a file: 03_streaming.py
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()

model = init_chat_model("gemini-3.5-flash", model_provider="google-genai")

print("Streaming response:\n")
for chunk in model.stream("Write a 3-sentence explanation of how vector databases work."):
    print(chunk.content, end="", flush=True)
print()  # newline at the end
