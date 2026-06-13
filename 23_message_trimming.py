# create a file: 23_message_trimming.py
from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, trim_messages
)
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# -------------------------------------------------------
# Simulate a long editorial session at Tech News Daily
# This is what accumulates after 30+ minutes of work
# -------------------------------------------------------
editorial_session = [
    SystemMessage(content="You are the senior editor at Tech News Daily. Help the team triage today's news queue."),
    HumanMessage(content="We have 47 articles in the queue. Where should we start?"),
    AIMessage(content="Start with the zero-day CVE story — it's time-sensitive. After that, triage by category: security first, then AI/cloud, then hardware."),
    HumanMessage(content="The CVE story is published. What's next?"),
    AIMessage(content="Move to the AWS GPU pricing story. It's been sitting for 2 hours and relates to the NVIDIA Blackwell story from yesterday."),
    HumanMessage(content="Should the AWS and NVIDIA stories run together as a package?"),
    AIMessage(content="Good idea. Commission a 200-word editor's note connecting the pricing pressure to the Blackwell announcement. Run the package at 11 AM."),
    HumanMessage(content="The NVIDIA story needs a quote from their PR team. We're waiting on approval."),
    AIMessage(content="Hold the NVIDIA piece. Run the AWS story standalone. When the quote comes in, publish NVIDIA separately with a cross-link."),
    HumanMessage(content="Quote approved. Can we still make the 11 AM window?"),
    AIMessage(content="Yes — add the quote to the final two paragraphs, update the metadata, and publish. The AWS story is already live so the cross-link will work immediately."),
    # ... dozens more turns in a real session
    HumanMessage(content="End of morning queue. What's the afternoon priority?"),
]

print(f"Full history: {len(editorial_session)} messages")

# -------------------------------------------------------
# Strategy 1: Trim to last N tokens
# Keeps the most recent messages that fit within the budget.
# The system message is always preserved (include_system=True).
# -------------------------------------------------------
trimmed_by_tokens = trim_messages(
    editorial_session,
    max_tokens=500,           # keep last ~500 tokens
    strategy="last",          # keep the most recent messages
    token_counter=llm,        # LangChain uses the model's tokeniser
    include_system=True,      # always keep the system prompt
    allow_partial=False,      # never cut a message in half
)
print(f"\nAfter token trim (max 500 tokens): {len(trimmed_by_tokens)} messages")
for m in trimmed_by_tokens:
    role = m.__class__.__name__.replace("Message", "")
    print(f"  {role}: {m.content[:60]}...")

# -------------------------------------------------------
# Strategy 2: Trim to last N messages
# Simpler but less precise — use when token counting is unavailable
# -------------------------------------------------------
trimmed_by_count = trim_messages(
    editorial_session,
    max_tokens=6,             # keep last 6 messages (used as count here)
    strategy="last",
    token_counter=len,        # count messages, not tokens
    include_system=True,
    allow_partial=False,
)
print(f"\nAfter message count trim (last 6): {len(trimmed_by_count)} messages")

# -------------------------------------------------------
# Integrate trimming directly into the chain
# -------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from functools import partial

def trim_for_model(messages, max_tokens=800):
    """Trim messages before they enter the model."""
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
    )

editorial_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages")
])

# The chain trims before the prompt is formatted
trimming_chain = trim_for_model | editorial_template | llm

response = trimming_chain.invoke({"messages": editorial_session})
print(f"\nResponse with trimming chain:")
print(response.content[:200])
