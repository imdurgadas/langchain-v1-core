# create a file: 09_vectorless_rag.py
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

document = """
Security Policy — Password Requirements

Section 1: Password Complexity
All user passwords must be a minimum of 12 characters in length.
Passwords must contain at least one uppercase letter, one lowercase letter,
one numeric digit, and one special character from the set: !@#$%^&*

Section 2: Password Rotation
All passwords must be rotated every 90 days.
Users will receive a reminder 7 days before expiry.
Three previous passwords cannot be reused.

Section 3: Multi-Factor Authentication
MFA is mandatory for all accounts with admin privileges.
Supported MFA methods: TOTP apps, hardware security keys.
SMS-based MFA is deprecated and will be removed in Q3 2026.
"""

# Step 1: Build the index — ask the LLM to parse the document structure
index_prompt = f"""Read this document and create a JSON index mapping each section title 
to its full content. Return valid JSON only, no explanation.

Format: {{"sections": [{{"title": "...", "content": "...", "keywords": [...]}}]}}

Document:
{document}"""

raw_index = llm.invoke(index_prompt).content
# Strip markdown code fences if the model wraps the JSON
raw_index = raw_index.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
index = json.loads(raw_index)

print("Built index with sections:")
for section in index["sections"]:
    print(f"  - {section['title']}")

# Step 2: Answer questions by navigating the index
def ask_vectorless(question: str):
    index_summary = "\n".join(
        f"Section: {s['title']} | Keywords: {', '.join(s.get('keywords', []))}"
        for s in index["sections"]
    )

    # First call: identify which section to retrieve
    routing_prompt = f"""Given this document index:
{index_summary}

Which section title is most relevant to answer: "{question}"?
Reply with ONLY the exact section title, nothing else."""

    target_section_title = llm.invoke(routing_prompt).content.strip()

    # Retrieve the matching section content
    section_content = next(
        (s["content"] for s in index["sections"] if s["title"].lower() == target_section_title.lower()),
        None
    )

    if not section_content:
        return "Could not locate a relevant section in the index."

    # Second call: answer from the retrieved section
    answer_prompt = f"""Use only the following content to answer the question.

Content:
{section_content}

Question: {question}"""

    return llm.invoke(answer_prompt).content

print("\n" + ask_vectorless("What are the password requirements?"))
print("\n" + ask_vectorless("When will I get notified before my password expires?"))
