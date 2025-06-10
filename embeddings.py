import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()  # now os.getenv(...) will pick up values from your .env file


def get_llm():
    """
    Returns a ChatGroq LLM instance (Llama 3.3 70B) using the GROQ API key
    stored in the environment.
    """
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY", "")  # Put your actual GROQ key in .env as GROQ_API_KEY
    )
    return llm

# ──────────────────────────────────────────────────────────────────────────────
# 1. Text Splitter (512 tokens per chunk, 100 token overlap)
# ──────────────────────────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Embeddings Model (HuggingFace BGE) on CPU
# ──────────────────────────────────────────────────────────────────────────────


HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

from huggingface_hub import login

login(HF_TOKEN)

model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
# ──────────────────────────────────────────────────────────────────────────────
# 3. Prompt Template for RAG Assistant
# ──────────────────────────────────────────────────────────────────────────────
prompt_template = """
You are an assistant specialized in analyzing and improving website performance. Your goal is to provide accurate, practical, and performance-driven answers.
Use the following retrieved context (such as PageSpeed Insights data or audit results) to answer the user's question.
If the context lacks sufficient information, respond with "I don't know." Do not make up answers or provide unverified information.

Guidelines:
1. Extract relevant performance insights from the context to form a helpful and actionable response.
2. Maintain a clear, professional, and user-focused tone.
3. If the question is unclear or needs more detail, ask for clarification politely.
4. Prioritize recommendations that follow web performance best practices (e.g., optimizing load times, reducing blocking resources, improving visual stability).

Retrieved context:
{context}

User's question:
{question}

Your response:
"""

user_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{question}"),
    ]
)
