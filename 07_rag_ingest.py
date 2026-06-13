# create a file: 07_rag_ingest.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os

# -------------------------------------------------------
# Component 1: Chunking
# -------------------------------------------------------
# Why RecursiveCharacterTextSplitter?
# It tries to split on natural boundaries first (\n\n, then \n, then spaces)
# before falling back to hard character limits. This preserves paragraph
# structure better than a fixed-size splitter.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,     # target chunk size in characters
    chunk_overlap=40,   # overlap between chunks to preserve context at boundaries
    separators=["\n\n", "\n", " ", ""]
)

# For this example, we simulate a document with raw text
# In production, replace this with PyMuPDFLoader for PDFs
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

chunks = splitter.create_documents([sample_document_text])
print(f"Document split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n{chunk.page_content}")
