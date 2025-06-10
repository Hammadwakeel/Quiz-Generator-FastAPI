from pydantic import BaseModel, Field
from typing import List, Optional

class IngestRequest(BaseModel):
    """
    Request body for ingesting documents into a user's FAISS vector store.
    """
    documents: List[str] = Field(
        ...,
        description="A list of text documents (strings) to ingest into the vector store."
    )

class IngestResponse(BaseModel):
    """
    Response after ingesting documents for a user.
    """
    success: bool
    message: str
    user_id: str
    vectorstore_path: Optional[str] = None

class CreateChatResponse(BaseModel):
    """
    Response after creating a new chat session for a user.
    """
    success: bool
    message: str
    user_id: str
    chat_id: Optional[str] = None

class ChatRequest(BaseModel):
    """
    Body for sending a user message to an existing chat session.
    """
    question: str = Field(..., description="The user's question or message.")

class ChatResponse(BaseModel):
    """
    Response from the RAG chatbot endpoint.
    """
    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    chat_id: str
    user_id: str
