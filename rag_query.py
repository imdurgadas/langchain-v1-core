# create a file: 08_rag_query.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- Setup (same as before) ---
sample_document_text = """
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

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = splitter.create_documents([sample_document_text])

# --- Embed and Store ---
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="security_policy",
    metadata={"hnsw:space": "cosine"}
    # cosine distance: measures angle between vectors, not magnitude
    # better than euclidean for text similarity
)

texts = [chunk.page_content for chunk in chunks]
embeddings = encoder.encode(texts).tolist()
ids = [f"chunk_{i}" for i in range(len(texts))]

collection.add(documents=texts, embeddings=embeddings, ids=ids)
print(f"Indexed {len(texts)} chunks into ChromaDB")

# --- Query ---
def ask(question: str):
    # Embed the question with the same model used for indexing
    query_vec = encoder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=2)
    retrieved_chunks = results["documents"][0]

    # Build the RAG prompt: retrieved context + the user's question
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question using ONLY the information provided below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
"""
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    response = llm.invoke(prompt)
    return response.content

# Test it
print("\n" + ask("What are the password complexity requirements?"))
print("\n" + ask("How long before my password expires will I be reminded?"))
print("\n" + ask("What is the company's vacation policy?"))  # should say "I don't have that"
