import time
from typing import List, Dict, Any
from pymongo import ReturnDocument

from config import settings
from db import mongo_client, chat_collection_name
from embeddings import get_llm
from langchain.prompts import ChatPromptTemplate
from logging_config import logger

# Get the actual collection object
db = mongo_client[settings.mongo_chat_db]
coll = db[chat_collection_name]

# LLM & summarization prompt
llm = get_llm()
summarization_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following conversation into a concise summary:"),
    ("human", "{chat_history}")
])

class ChatHistoryManager:
    @staticmethod
    def create_session(chat_id: str) -> None:
        """Ensure a document exists for this chat_id with empty messages."""
        coll.update_one(
            {"session_id": chat_id},
            {"$setOnInsert": {"session_id": chat_id, "messages": []}},
            upsert=True
        )
        logger.info("Initialized chat session %s", chat_id)

    @staticmethod
    def get_messages(chat_id: str) -> List[Dict[str, Any]]:
        """Return the messages array for this session (or empty if none)."""
        doc = coll.find_one({"session_id": chat_id}, {"_id": 0, "messages": 1})
        return doc.get("messages", []) if doc else []

    @staticmethod
    def add_message(chat_id: str, role: str, content: str) -> None:
        """Append a new {role,content,timestamp} entry to the messages array."""
        entry = {
            "type": role,
            "content": content,
            "timestamp": time.time()
        }
        coll.update_one(
            {"session_id": chat_id},
            {"$push": {"messages": entry}}
        )
        logger.debug("Appended %s message to %s", role, chat_id)

    @staticmethod
    def summarize_if_needed(chat_id: str, threshold: int = 10) -> bool:
        """
        If message count > threshold, summarize and replace all messages
        with a single "ai" summary entry.
        """
        messages = ChatHistoryManager.get_messages(chat_id)
        if len(messages) <= threshold:
            return False

        # Flatten for summarization
        chat_text = "\n".join(f"{m['type'].upper()}: {m['content']}" for m in messages)

        # Run summarization
        summary_chain = summarization_prompt | llm
        result = summary_chain.invoke({"chat_history": chat_text})
        summary = getattr(result, "content", result)

        # Replace entire messages array with the summary
        coll.find_one_and_update(
            {"session_id": chat_id},
            {"$set": {"messages": [
                {"type": "ai", "content": summary, "timestamp": time.time()}
            ]}},
            return_document=ReturnDocument.AFTER
        )
        logger.info("Summarized chat %s down to one message", chat_id)
        return True

    @staticmethod
    def get_retrieved_context(chat_id: str) -> str:
        """
        Return concatenated human messages from chat history for context retrieval.
        """
        messages = ChatHistoryManager.get_messages(chat_id)
        context = "\n".join([
            m["content"] for m in messages
            if m["type"] == "human" or m["type"] == "user"
        ])
        logger.debug("Retrieved context for %s: %s", chat_id, context[:100])
        return context
