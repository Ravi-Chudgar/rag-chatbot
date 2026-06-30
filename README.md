# RAG Chatbot (FastAPI + Ollama + FAISS)

This project is a Retrieval-Augmented Generation (RAG) chatbot with login, per-user chat history, and admin history view.

## Features

- Ingest `.txt`, `.md`, and `.pdf` documents
- Chunk and embed text with Ollama embeddings
- Store vectors in local FAISS index
- Ask questions against indexed document context
- Return source chunks used to answer
- Login and register users
- Save chat history per user in SQLite
- Admin dashboard to view all users' chat history

## Project Structure

```text
rag-chatbot/
├── app/
│   ├── config.py
│   ├── embeddings.py
│   ├── ingest.py
│   ├── main.py
│   ├── models.py
│   ├── rag.py
│   ├── utils.py
│   └── vector_store.py
├── data/                  # put source docs here
├── vector_db/             # generated FAISS index
├── .env.example
├── requirements.txt
└── run.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Make sure Ollama is running and pull models:

```bash
ollama serve
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Run

```bash
python run.py
```

Web UI will be available at `http://127.0.0.1:8000` and API at the same host.

Default admin credentials (change in `.env`):

- Username: `admin`
- Password: `admin123`

## Endpoints

- `GET /` -> Web UI
- `GET /login` -> login/register page
- `GET /chat-ui` -> authenticated chat UI
- `GET /admin-ui` -> admin-only history UI
- `GET /health` -> health check

### 1) Ingest all files from `data/`

```bash
curl -X POST "http://127.0.0.1:8000/ingest-directory"
```

### 2) Ingest one uploaded file

```bash
curl -X POST "http://127.0.0.1:8000/ingest-file" \
  -F "file=@/absolute/path/to/file.pdf"
```

### 3) Chat

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?"}'
```