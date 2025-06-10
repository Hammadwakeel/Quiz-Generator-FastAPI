**Project Name: AI Quiz Generator API**

An easy-to-use FastAPI service that lets users upload PDF/DOCX source documents and automatically generates quizzes (MCQs, True/False, Short Answer) based strictly on the uploaded content.

---

## Features

* **Document Ingestion**: Upload one or more PDF or DOCX files.
* **Chunking & Embeddings**: Splits text into semantic chunks and embeds via HuggingFace/FAISS.
* **Context-Aware Quiz Generation**: Uses a LangChain quiz-making chain with a custom prompt template to generate quizzes that match user-specified type, difficulty, and number of questions.
* **MongoDB-Backed Chat History**: Stores and summarizes prior requests to maintain context.
* **Single Endpoint Quiz Flow**: `/rag/chat/{user_id}/{chat_id}` generates a quiz—no regular chat chain is involved.

---

## Tech Stack

* **Backend Framework**: FastAPI
* **Vector Store**: FAISS via `langchain_community`
* **Embeddings**: HuggingFace BGE
* **Memory & RAG**: LangChain
* **Database**: MongoDB (for chat history)
* **Document Loaders**: `PyPDFLoader`, `Docx2txtLoader`
* **Logging**: Standard Python `logging`

---

## Prerequisites

* Python 3.10+
* MongoDB Instance (local or cloud)
* (Optional) Virtual environment tool (venv, conda)

---

## Installation

1. **Clone Repository**

   ```bash
   git clone https://github.com/Hammadwakeel/Quiz-Generator-FastAPI.git
   cd Quiz-Generator-FastAPI
   ```

2. **Create & Activate Virtualenv**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/Mac  
   .venv\Scripts\activate         # Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   Create a `.env` file in the project root:

   ```dotenv
   MONGO_URI=mongodb://localhost:27017
   MONGO_CHAT_DB=quiz_chat_db
   MONGO_CHAT_COLLECTION=chat_sessions
   HUGGINGFACE_API_KEY=your_hf_api_key
   ```

---

## Running the Server

```bash
uvicorn app.rag.routes:router --reload
```

By default the API will be available at `http://127.0.0.1:8000/rag`.

Visit **`/docs`** for interactive Swagger UI.

---

## API Endpoints

### 1. Ingest Documents

**POST** `/rag/ingest/{user_id}`
Upload one or more `.pdf` or `.docx` files to build a FAISS vectorstore.

* **Path Parameters**

  * `user_id` (string)

* **Form Data**

  * `files`: one or more `UploadFile` objects

* **Response**

  ```json
  {
    "success": true,
    "message": "Vectorstore created successfully.",
    "user_id": "hammad",
    "vectorstore_path": "/path/to/faiss/index"
  }
  ```

### 2. Create Chat Session

**POST** `/rag/chat/create/{user_id}`
Generates a new `chat_id` and initializes history in MongoDB.

* **Path Parameters**

  * `user_id` (string)

* **Response**

  ```json
  {
    "success": true,
    "message": "Chat session created.",
    "user_id": "hammad",
    "chat_id": "e9ac1349-e800-4ebb-b5fb-9a0ac6ea6b17"
  }
  ```

### 3. Generate Quiz

**POST** `/rag/chat/{user_id}/{chat_id}`
Generates a quiz from the previously ingested documents.

* **Path Parameters**

  * `user_id` (string)
  * `chat_id` (UUID)

* **Request Body**

  ```jsonc
  {
    "question": "Create quiz on machine learning with 10 questions and hard complexity MCQs type."
  }
                    
  ```

* **Response**

  ```json
  {
    "success": true,
    "answer": "1. What is Machine Learning ...",
    "error": null,
    "chat_id": "e9ac1349-e800-4ebb-b5fb-9a0ac6ea6b17",
    "user_id": "hammad"
  }
  ```

---

## Project Structure

```
.
├──__init__.py
├── chat_history.py
├── config.py
├── db.py
├── embeddings.py               
├── logging_config.py
├── main.py
├── routes.py
├── schemas.py
├── utils.py
├── requirements.txt
├── .gitignore
├── .env
└── README.md
```

---

## Example Usage with cURL

1. **Ingest**

   ```bash
   curl -X POST "http://localhost:8000/rag/ingest/alice" \
     -F "files=@source.pdf" -F "files=@notes.docx"
   ```

2. **Create Session**

   ```bash
   curl -X POST "http://localhost:8000/rag/chat/create/alice"
   ```

3. **Generate Quiz**

   ```bash
   curl -X POST "http://localhost:8000/rag/chat/alice/<chat_id>" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Create a quiz on Machine Learning. Quiz type should be MCQs and complexity should be Hard and Total number of questions should be 10"
     }'
   ```

---
