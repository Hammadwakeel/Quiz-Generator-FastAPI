import os
import uuid
from fastapi import APIRouter, HTTPException
from typing import Optional

from schemas import (
    IngestRequest,
    IngestResponse,
    CreateChatResponse,
    ChatRequest,
    ChatResponse
)
from utils import (
    text_splitter,
    embeddings,
    get_vectorstore_path,
    save_vectorstore_to_disk,
    upsert_vectorstore_metadata,
    build_or_load_vectorstore,
    build_rag_chain,
    initialize_chat_history
)
from logging_config import logger

from chat_history import ChatHistoryManager

router = APIRouter(prefix="/rag", tags=["rag"])

from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS as _FAISS

@router.post("/ingest/{user_id}", response_model=IngestResponse)
async def ingest_documents(user_id: str, body: IngestRequest):
    """
    Ingest a mix of PDF paths, DOCX paths, or raw-text strings into a FAISS vectorstore.
    """
    logger.info("Ingestion requested for user_id=%s. Num inputs=%d", user_id, len(body.documents))
    try:
        # 1. Load / extract text from each input
        all_texts = []
        for entry in body.documents:
            path = entry.strip()
            if path.lower().endswith(".pdf"):
                loader = PyPDFLoader(path, mode="page")
                docs = loader.load()
                all_texts.extend([d.page_content for d in docs])
                logger.debug("Loaded %d pages from PDF %s", len(docs), path)

            elif path.lower().endswith(".docx"):
                loader = Docx2txtLoader(path)
                docs = loader.load()
                all_texts.extend([d.page_content for d in docs])
                logger.debug("Loaded %d elements from DOCX %s", len(docs), path)

            else:
                # treat as raw text
                all_texts.append(path)
                logger.debug("Treated raw text input of length %d", len(path))

        if not all_texts:
            raise HTTPException(status_code=400, detail="No valid documents or text provided")

        # 2. Concatenate all extracted text
        all_text = "\n\n".join(all_texts)

        # 3. Split into chunks
        text_chunks = text_splitter.split_text(all_text)
        logger.info("Split into %d chunks", len(text_chunks))

        # 4. Build FAISS vectorstore
        vs = _FAISS.from_texts(texts=text_chunks, embedding=embeddings)

        # 5. Save to disk
        faiss_path = save_vectorstore_to_disk(vs, user_id)
        logger.info("Saved FAISS index to %s", faiss_path)

        # 6. Upsert metadata
        upsert_vectorstore_metadata(user_id, faiss_path)
        logger.info("Upserted vectorstore metadata for user_id=%s", user_id)

        return IngestResponse(
            success=True,
            message="Vectorstore created successfully.",
            user_id=user_id,
            vectorstore_path=faiss_path
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during ingestion for user_id=%s: %s", user_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@router.post("/chat/create/{user_id}", response_model=CreateChatResponse)
async def create_chat_session(user_id: str):
    """
    Create a new chat session for this user:
      - Generate a chat_id (UUID).
      - Initialize an empty MongoDBChatMessageHistory for that chat_id.
      - Return the chat_id so the client can use it in subsequent calls.
    """
    logger.info("Creating new chat session for user_id=%s", user_id)
    try:
        chat_id = str(uuid.uuid4())

        # Initialize chat history (this writes an empty session to Mongo)
        _ = initialize_chat_history(chat_id)
        logger.info("Created chat history in Mongo for chat_id=%s", chat_id)

        return CreateChatResponse(
            success=True,
            message="Chat session created.",
            user_id=user_id,
            chat_id=chat_id
        )
    except Exception as e:
        logger.error("Error creating chat for user_id=%s: %s", user_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create chat session: {e}")


@router.post("/chat/{user_id}/{chat_id}", response_model=ChatResponse)
async def chat_with_user(user_id: str, chat_id: str, body: ChatRequest):
    question = body.question.strip()
    logger.info("Chat request user=%s chat=%s question=%s", user_id, chat_id, question)

    try:
        # 1) Ensure session exists
        ChatHistoryManager.create_session(chat_id)

        # 2) Summarize long histories
        ChatHistoryManager.summarize_if_needed(chat_id, threshold=10)

        # 3) Record the user message
        ChatHistoryManager.add_message(chat_id, role="human", content=question)

        # 4) Build and invoke the RAG chain
        chain = build_rag_chain(user_id, chat_id)
        history = ChatHistoryManager.get_messages(chat_id)
        result = chain.invoke({"question": question, "chat_history": history})
        answer = result.get("answer") or result.get("output_text")
        if not answer:
            raise Exception("No answer returned from chain")

        # 5) Record the AI response
        ChatHistoryManager.add_message(chat_id, role="ai", content=answer)

        return ChatResponse(
            success=True,
            answer=answer,
            error=None,
            chat_id=chat_id,
            user_id=user_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error chatting user=%s chat=%s: %s", user_id, chat_id, e, exc_info=True)
        return ChatResponse(
            success=False,
            answer=None,
            error=str(e),
            chat_id=chat_id,
            user_id=user_id
        )

@router.get("/")
async def Welcome():
    """
    Welcome endpoint to verify the API is running.
    """
    return {"message": "Welcome to the RAG API! Use /rag/ingest to upload documents or /rag/chat to start chatting."}
