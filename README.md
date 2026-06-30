# RAG Chatbot (FastAPI + Multi-Provider LLM + FAISS)

This project is a Retrieval-Augmented Generation (RAG) chatbot with support for multiple LLM providers, login, per-user chat history, and admin history view.

## Features

- **Multi-Provider LLM Support**: Choose from Ollama, OpenAI, Claude (Anthropic), or HuggingFace
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
│   ├── config.py           # Configuration management
│   ├── embeddings.py
│   ├── ingest.py
│   ├── llm_factory.py      # Multi-provider LLM factory
│   ├── main.py
│   ├── models.py
│   ├── rag.py
│   ├── utils.py
│   └── vector_store.py
├── data/                   # put source docs here
├── vector_db/              # generated FAISS index
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

## Configuration

Edit `.env` and set `LLM_PROVIDER` to one of: `ollama`, `openai`, `claude`, or `huggingface`.

### Using Ollama (Default)

```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Make sure Ollama is running:

```bash
ollama serve
ollama pull llama3.1
ollama pull nomic-embed-text
```

### Using OpenAI

```bash
# .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-3.5-turbo  # or gpt-4, etc.
```

### Using Claude (Anthropic)

```bash
# .env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_CHAT_MODEL=claude-3-sonnet-20240229  # or claude-3-opus, etc.
```

### Using HuggingFace

```bash
# .env
LLM_PROVIDER=huggingface
HUGGINGFACE_API_KEY=hf_...
HUGGINGFACE_CHAT_MODEL=HuggingFaceH4/zephyr-7b-beta
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