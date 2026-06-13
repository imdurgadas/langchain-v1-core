# create a file: 24_summarisation_fallback.py
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# -------------------------------------------------------
# Build a summariser that compresses old conversation turns
# into a single context-preserving message
# -------------------------------------------------------
def summarise_history(messages: list, keep_last_n: int = 4) -> list:
    """
    Splits messages into 'old' and 'recent'.
    Summarises the old part and prepends the summary to the recent part.
    
    Why not summarise everything?
    The most recent N messages are the ones directly relevant to the
    current task. Summarising them loses the precision you need right now.
    Keep them verbatim and only compress the older context.
    """
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    
    if len(non_system) <= keep_last_n:
        return messages  # nothing old enough to compress
    
    old_messages = non_system[:-keep_last_n]
    recent_messages = non_system[-keep_last_n:]
    
    # Build a summary request for the old messages
    old_text = "\n".join([
        f"{m.__class__.__name__.replace('Message', '')}: {m.content}"
        for m in old_messages
    ])
    
    summary_prompt = f"""Summarise the following conversation history in 3–5 bullet points.
Preserve all factual decisions, named entities, and action items.
This summary will be used as context for continuing the conversation.

Conversation to summarise:
{old_text}"""
    
    summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
    
    # Package the summary as an AI message so it slots naturally into history
    compressed = AIMessage(
        content=f"[CONTEXT SUMMARY — prior {len(old_messages)} messages compressed]\n{summary_response.content}"
    )
    
    return system_messages + [compressed] + recent_messages


# -------------------------------------------------------
# Test with the newsroom session from Part 1
# -------------------------------------------------------
newsroom_session = [
    SystemMessage(content="You are the senior editor at Tech News Daily."),
    HumanMessage(content="We have 47 articles in the queue today."),
    AIMessage(content="Start with the CVE zero-day story. Publish within the hour."),
    HumanMessage(content="Done. The AWS GPU pricing story is next?"),
    AIMessage(content="Yes. Package it with the NVIDIA Blackwell piece at 11 AM."),
    HumanMessage(content="NVIDIA PR team hasn't responded yet."),
    AIMessage(content="Hold NVIDIA. Run AWS standalone and cross-link when NVIDIA is ready."),
    HumanMessage(content="AWS is live. NVIDIA quote just came in."),
    AIMessage(content="Add the quote to the final two paragraphs. Publish NVIDIA immediately."),
    HumanMessage(content="Both are live. What's the afternoon focus?"),  # ← current turn
]

print(f"Original: {len(newsroom_session)} messages")

compressed_session = summarise_history(newsroom_session, keep_last_n=4)
print(f"After summarisation: {len(compressed_session)} messages")

# Show the compressed context
for m in compressed_session:
    role = m.__class__.__name__.replace("Message", "")
    print(f"\n{role}:")
    print(m.content[:300])

# -------------------------------------------------------
# Decision logic: when to trim vs when to summarise
# -------------------------------------------------------
def manage_context(messages: list, token_budget: int = 2000) -> list:
    """
    Adaptive context manager:
    - Under budget → pass through unchanged
    - Over budget + history is factual (research) → summarise
    - Over budget + history is conversational → trim
    
    In production, you would determine 'factual vs conversational'
    from metadata on each session (e.g., session_type="research").
    Here we check whether any message is longer than 200 chars
    as a heuristic for factual content.
    """
    from langchain_core.messages import trim_messages
    
    # Rough token estimate (4 chars ≈ 1 token)
    total_chars = sum(len(m.content) for m in messages)
    estimated_tokens = total_chars // 4
    
    if estimated_tokens <= token_budget:
        return messages  # no action needed
    
    # Heuristic: if any message > 200 chars, treat as factual/research session
    has_long_messages = any(len(m.content) > 200 for m in messages)
    
    if has_long_messages:
        print("  → Summarising (factual/research session)")
        return summarise_history(messages, keep_last_n=4)
    else:
        print("  → Trimming (conversational session)")
        return trim_messages(messages, max_tokens=token_budget, 
                           strategy="last", token_counter=len,
                           include_system=True, allow_partial=False)

print("\n\n=== Adaptive Context Manager ===")
result = manage_context(newsroom_session, token_budget=100)
print(f"Output: {len(result)} messages")
