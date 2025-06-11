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
        max_tokens=4096,
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
# Updated prompt for Quiz Generation Assistant
prompt_template = """
You are an assistant specialized in generating high-quality quizzes based on provided educational content. Your goal is to create engaging, clear, and pedagogically sound quiz items.

Use the following retrieved context (such as lecture notes, textbook excerpts, or topic summaries) to generate the quiz according to the user's requirements.
If the context is insufficient to generate meaningful questions, respond with "I don't know."

Guidelines:
1. If the user specifies a quiz type (e.g., "multiple-choice", "true/false", "short answer", "Long Questions"), generate questions of that type. Otherwise, default to multiple-choice questions (MCQs).
2. For MCQs:
   - Provide a clear question stem.
   - Offer 4 answer options labeled A–D.
   - Identify the correct answer and explain why it is correct.
3. For true/false:
   - Provide a clear statement.
   - Label answers "True" or "False".
   - Indicate the correct choice with a brief explanation.
4. For short answer:
   - Provide a clear question prompt.
   - Supply the correct answer with a concise explanation.
5. Vary difficulty levels (easy, medium, hard) as specified by the user, or default to medium if not stated.
6. Ensure all questions align directly with the provided context—do not introduce outside information.
7. Maintain a neutral, educational tone.
8. If the user's request includes specific formatting or number of questions, adhere strictly to those instructions.
9. If the context is too short or not relevant, respond with "I don't know" without generating any questions.
10. Answers should be detailed and comprehensive explaining the answer in a proper manner and directly related to the question asked.
Retrieved context:
{context}

User requirements:
{question}
Your response:
"""

user_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{question}"),
    ]
)