# create a file: 25_context_layers.py
"""
The 5 Layers of Agent Context
==============================

Layer 1 — STATIC SYSTEM CONTEXT
  What it is: the system prompt, role definition, core constraints
  Changes: never (or very rarely — version controlled)
  Token budget: ≤ 300 tokens
  Why: this is the agent's "identity" — it must always be present

Layer 2 — RUNTIME CONTEXT  
  What it is: session metadata injected at invocation time
  Examples: current date, user role, active project name, feature flags
  Changes: once per session
  Token budget: ≤ 100 tokens
  Why: avoids hardcoding time-sensitive values into the system prompt

Layer 3 — TASK CONTEXT
  What it is: the specific task details for this invocation
  Examples: the article to classify, the research brief, the draft to review
  Changes: once per task
  Token budget: varies — this is the main payload
  Why: keeps task data separate from session data; easier to test

Layer 4 — RETRIEVED CONTEXT
  What it is: documents fetched by RAG (from Part 3) relevant to this task
  Examples: past articles on this topic, style guide excerpts
  Changes: per task, fetched dynamically
  Token budget: ≤ 20% of total window
  Why: RAG context is high-value but can be large — budget it explicitly

Layer 5 — CONVERSATION HISTORY
  What it is: the multi-turn dialogue (managed by Parts 1–2 above)
  Changes: every turn
  Token budget: what's left after Layers 1–4
  Why: history is the most variable layer — apply trimming/summarisation here
"""

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import datetime

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

def build_layered_context(
    task_content: str,
    retrieved_docs: list[str],
    conversation_history: list,
    user_role: str = "editor"
) -> list:
    """
    Assembles the full context window in deliberate order.
    Each layer has a defined purpose and token budget.
    """
    messages = []
    
    # Layer 1: Static system context (≤ 300 tokens)
    messages.append(SystemMessage(content="""You are the editorial AI assistant for Tech News Daily.
Core rules:
- Prioritise accuracy over speed
- Flag uncertain facts with [UNVERIFIED]
- Never publish unverified breaking news claims
- Cross-link related stories when relevant"""))
    
    # Layer 2: Runtime context (≤ 100 tokens — injected once per session)
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    messages.append(SystemMessage(content=f"""Session context:
Date/time: {now}
User role: {user_role}
Edition: Morning Queue
Active stories: AWS GPU pricing (published), NVIDIA Blackwell (published)"""))
    
    # Layer 3: Task context — the current work item
    messages.append(HumanMessage(content=f"Current task:\n{task_content}"))
    
    # Layer 4: Retrieved context from RAG (injected as AI context note)
    if retrieved_docs:
        retrieved_text = "\n\n".join([f"[Related article]: {doc}" for doc in retrieved_docs])
        messages.append(AIMessage(content=f"""[RETRIEVED CONTEXT — for reference only, not part of conversation]
{retrieved_text}"""))
    
    # Layer 5: Conversation history (trimmed, added last)
    messages.extend(conversation_history)
    
    return messages


# -------------------------------------------------------
# Demonstrate the layered assembly
# -------------------------------------------------------
task = "Review this draft headline: 'Google Cuts Cloud Prices Again — Third Time in Six Months'. Is the claim accurate? Should we add context?"

related_articles = [
    "Google Cloud announced a 15% price reduction on Cloud Storage in January 2026.",
    "A second Google Cloud pricing update in April 2026 reduced Compute Engine costs by 8%.",
]

prior_conversation = [
    HumanMessage(content="What headline style does Tech News Daily use for pricing stories?"),
    AIMessage(content="We use factual, numeric headlines. Include the percentage and the service name. Avoid superlatives."),
]

context = build_layered_context(
    task_content=task,
    retrieved_docs=related_articles,
    conversation_history=prior_conversation,
    user_role="senior_editor"
)

print(f"Total messages in context: {len(context)}")
for i, m in enumerate(context):
    role = m.__class__.__name__.replace("Message", "")
    layer = ["Static", "Runtime", "Task", "Retrieved", "History"][min(i, 4)]
    print(f"  [{layer}] {role}: {m.content[:80]}...")

response = llm.invoke(context)
print(f"\nAssistant response:")
print(response.content)
