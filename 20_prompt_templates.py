# create a file: 20_prompt_templates.py
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# -------------------------------------------------------
# Basic template: named variables replace f-string interpolation
# -------------------------------------------------------
headline_template = ChatPromptTemplate.from_messages([
    ("system", """You are a senior editor at Tech News Daily.
Your job is to classify incoming news articles and extract clean metadata.
Be precise. Do not add commentary outside the requested fields."""),
    ("human", """Article title: {title}
Article body: {body}

Classify this article and extract: category, estimated reading time (minutes), and a 1-sentence summary.""")
])

# The template is now an object you can inspect, test, and store
print("Template variables:", headline_template.input_variables)
# → ['title', 'body']

# Invoke by passing variables — the template handles interpolation
chain = headline_template | llm

result = chain.invoke({
    "title": "Gemini 3.5 Flash Cuts Inference Costs by 40%",
    "body": "Google DeepMind announced today that Gemini 3.5 Flash achieves a 40% reduction in per-token inference costs compared to its predecessor, while maintaining benchmark parity on coding and reasoning tasks. The model is available immediately via the Gemini API."
})
print("\nRaw response (still a string):")
print(result.content)
