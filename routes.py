import os
import shutil
import uuid
from fastapi import APIRouter, HTTPException, Body,UploadFile, File    # ← added Body here

from typing import List
import tempfile
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
from langchain.prompts import PromptTemplate
from embeddings import get_llm


router = APIRouter(prefix="/rag", tags=["rag"])

from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS as _FAISS

@router.post("/ingest/{user_id}", response_model=IngestResponse)
async def ingest_documents(
    user_id: str,
    files: List[UploadFile] = File(...),
):
    """
    Ingest uploaded PDF or DOCX files into a FAISS vectorstore.
    """
    # 1. Extract text from each uploaded file
    all_texts = []
    for upload in files:
        filename = upload.filename
        suffix = os.path.splitext(filename)[1].lower()
        # Save upload to temporary file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(upload.file, tmp)
                tmp_path = tmp.name
        finally:
            upload.file.close()

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path, mode="page")
                docs = loader.load()
                all_texts.extend([d.page_content for d in docs])

            elif suffix == ".docx":
                loader = Docx2txtLoader(tmp_path)
                docs = loader.load()
                all_texts.extend([d.page_content for d in docs])

            else:
                # unsupported file type
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
        finally:
            # Clean up temp file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if not all_texts:
        raise HTTPException(status_code=400, detail="No valid documents uploaded.")

    # 2. Concatenate all extracted text
    combined_text = "\n\n".join(all_texts)

    # 3. Split into chunks
    text_chunks = text_splitter.split_text(combined_text)

    # 4. Build FAISS vectorstore
    vs = _FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    # 5. Save to disk
    faiss_path = save_vectorstore_to_disk(vs, user_id)

    # 6. Upsert metadata
    upsert_vectorstore_metadata(user_id, faiss_path)

    return IngestResponse(
        success=True,
        message="Vectorstore created successfully.",
        user_id=user_id,
        vectorstore_path=faiss_path
    )

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
    
def recommend_courses(course: str, marks: float) -> str:
    """
    Recommend three next-step courses based on the completed `course` and `marks`.
    """
    template = """
    You are an academic advisor. A student just completed the course "{course}"
    with a score of {marks}%. Recommend three specific next-step university
    or online courses, and briefly explain why each is a good fit given their performance.
    in the output only provide the course names, separated by commas.
    Do not include any other text or explanations.
    Example output:
    "Advanced Physics, Data Science Fundamentals, Machine Learning Basics"
    """
    prompt = PromptTemplate.from_template(template)
    prompt_text = prompt.format(course=course, marks=marks)

    llm = get_llm()
    try:
        # invoke() may vary depending on your LLM wrapper (e.g. .generate, .predict, etc.)
        return llm.invoke(prompt_text)
    except Exception as e:
        logger.error(f"Course recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not generate course recommendations.")

@router.post("/recommendations")
async def get_course_recommendations(
    course: str = Body(..., description="Name of the completed course"),
    marks: float = Body(..., description="Score achieved in that course (percentage)")
):
    """
    Endpoint: POST /rag/recommendations
    Body JSON:
    {
      "course": "Physics 101",
      "marks": 85.5
    }
    Returns LLM-generated list of three recommended courses.
    """
    recommendations = recommend_courses(course, marks)
    return {"recommendations": recommendations}

@router.get("/")
async def Welcome():
    """
    Welcome endpoint to verify the API is running.
    """
    return {"message": "Welcome to the RAG API! Use /rag/ingest to upload documents or /rag/chat to start chatting."}
