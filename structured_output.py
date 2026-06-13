# create a file: 21_structured_output.py
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# -------------------------------------------------------
# Step 1: Define the schema as a Pydantic model
# This becomes the "contract" between your agent and its callers
# -------------------------------------------------------
class Article(BaseModel):
    """Structured representation of a news article's metadata."""
    
    headline: str = Field(
        description="The article headline, max 12 words, no question marks"
    )
    category: Literal["ai", "cloud", "security", "hardware", "business"] = Field(
        description="The primary category of the article"
    )
    summary: str = Field(
        description="A single sentence (max 25 words) summarising the article"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Your confidence in this classification, 0.0–1.0"
    )
    breaking: bool = Field(
        description="True if this is breaking news (announced in last 6 hours)"
    )

# -------------------------------------------------------
# Step 2: Bind the schema to the LLM
# Under the hood, LangChain chooses tool-calling vs JSON mode
# based on what the model supports — you do not need to decide
# -------------------------------------------------------
structured_llm = llm.with_structured_output(Article)

# -------------------------------------------------------
# Step 3: Build the chain — same pattern as before
# -------------------------------------------------------
classify_template = ChatPromptTemplate.from_messages([
    ("system", """You are a news classification engine for Tech News Daily.
Classify articles accurately. Confidence reflects how clearly the article fits one category.
If an article spans multiple categories, pick the most dominant one."""),
    ("human", """Article to classify:
Title: {title}
Body: {body}
Published: {published_at}""")
])

classify_chain = classify_template | structured_llm

# -------------------------------------------------------
# Test with three different articles
# -------------------------------------------------------
test_articles = [
    {
        "title": "AWS Announces 30% Price Cut on EC2 GPU Instances",
        "body": "Amazon Web Services reduced pricing on its P4d and P5 GPU instance families by 30% effective today, citing improved manufacturing economics and competition from Google Cloud and Azure in the AI training market.",
        "published_at": "2026-06-13T09:00:00Z"
    },
    {
        "title": "Critical Zero-Day Found in OpenSSH 9.x",
        "body": "Security researchers at Qualys disclosed a critical remote code execution vulnerability in OpenSSH versions 9.0–9.8. The CVE-2026-0412 flaw allows unauthenticated attackers to execute arbitrary code as root on vulnerable Linux systems. A patch is available.",
        "published_at": "2026-06-13T07:30:00Z"  # recent → breaking
    },
    {
        "title": "NVIDIA Unveils Blackwell Ultra B300 GPU Architecture",
        "body": "NVIDIA's new Blackwell Ultra architecture delivers 2.5x the FP8 throughput of H100 and introduces a new NVLink 5 interconnect. The B300 will ship in Q3 2026 with a focus on large-scale LLM inference clusters.",
        "published_at": "2026-06-12T14:00:00Z"
    }
]

print("=== Tech News Daily — Article Classifier ===\n")
for article in test_articles:
    result: Article = classify_chain.invoke(article)
    
    # result is a proper Python object — fully typed
    print(f"Title:      {article['title']}")
    print(f"Headline:   {result.headline}")
    print(f"Category:   {result.category}")
    print(f"Breaking:   {'🔴 YES' if result.breaking else 'No'}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Summary:    {result.summary}")
    print("-" * 60)
