# create a file: 22_output_parsers.py
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# -------------------------------------------------------
# StrOutputParser: the simplest parser — strips metadata,
# returns just the string content. Use this for free-form output.
# -------------------------------------------------------
summary_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a copy editor. Write tight, punchy summaries."),
        ("human", "Summarise this in exactly 2 sentences: {text}")
    ])
    | llm
    | StrOutputParser()  # converts AIMessage to plain string
)

text = "The EU AI Act's enforcement provisions came into effect today, requiring all high-risk AI systems deployed in the EU to undergo mandatory conformity assessments. Companies have 6 months to comply or face fines of up to 3% of global annual revenue."
summary = summary_chain.invoke({"text": text})
print("Summary (str):", summary)
print("Type:", type(summary))  # → <class 'str'>

# -------------------------------------------------------
# PydanticOutputParser: instructs the model via prompt to
# produce JSON matching your schema, then parses it.
# Less reliable than with_structured_output() but works
# with any model, including those without tool calling.
# -------------------------------------------------------
class EditDecision(BaseModel):
    """An editorial decision on a news article."""
    publish: bool = Field(description="Whether to publish this article")
    reason: str = Field(description="One sentence explaining the decision")
    priority: Literal["top", "standard", "hold"] = Field(
        description="Editorial priority: top (homepage), standard, or hold"
    )

parser = PydanticOutputParser(pydantic_object=EditDecision)

editorial_template = ChatPromptTemplate.from_messages([
    ("system", "You are the editor-in-chief of Tech News Daily."),
    ("human", """Evaluate this article and make an editorial decision.

Article: {article}

{format_instructions}""")
]).partial(format_instructions=parser.get_format_instructions())

editorial_chain = editorial_template | llm | parser

decision: EditDecision = editorial_chain.invoke({
    "article": "AWS quietly removed two deprecated EC2 instance types from its pricing page. No announcement was made."
})

print(f"\nPublish: {decision.publish}")
print(f"Priority: {decision.priority}")
print(f"Reason: {decision.reason}")

# -------------------------------------------------------
# OutputFixingParser: wraps any parser and adds an automatic
# retry if the initial parse fails.
# -------------------------------------------------------
# Simulate a scenario where the model might return malformed output:
fixing_parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=EditDecision),
    llm=llm
)

# In production, wrap your parser with OutputFixingParser for any
# model that sometimes produces slightly malformed JSON.
# The fixing parser calls the LLM once more with the parse error
# and the bad output, asking it to correct the format.
print("\n✓ OutputFixingParser configured — will auto-retry on malformed output")
print("  Use it like: fixing_parser.parse(malformed_string)")
