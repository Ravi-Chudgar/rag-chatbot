# RAG Chatbot Flow Diagram

## Document Ingestion Pipeline
```
📄 Upload Document (PDF/TXT/MD)
       ↓
📊 Text Extraction & Chunking
       ↓
🔢 Generate Embeddings (Ollama/OpenAI)
       ↓
🗄️ Store in FAISS Vector Database
       ↓
✅ Index Ready for Querying
```

## Chat Query Pipeline
```
❓ User Question
       ↓
🔐 Authentication Check (JWT)
       ↓
🔍 Semantic Search in Vector DB
       ↓
📚 Retrieve Top-K Relevant Documents
       ↓
🤖 Send to LLM with Context
       ↓
       ├─ Provider: OpenAI, Claude, Ollama, or HuggingFace
       ├─ Model: Configurable per provider
       └─ Temperature: 0 (deterministic)
       ↓
💬 Generate Answer with Sources
       ↓
💾 Save to Chat History (SQLite)
       ↓
📤 Return Response + Source Chunks
```

## Multi-Provider LLM Architecture
```
┌─────────────────────────────────────┐
│   LLM Factory (llm_factory.py)      │
└─────────────────────────────────────┘
                 ↓
    ┌────────────┼────────────┐
    ↓            ↓            ↓
 OLLAMA      OPENAI       CLAUDE
    ↓            ↓            ↓
Local      ChatGPT       Claude API
Models     (GPT-4)    (Sonnet/Opus)
    
    ↓            ↓            ↓
    └────────────┼────────────┘
                 ↓
          RAG Service
```

## Admin Dashboard Flow
```
👤 Admin Login
       ↓
🔓 Verify Credentials
       ↓
📊 Query SQLite Database
       ↓
📈 Display All Users' Chat History
       ↓
🔎 Search & Filter Options
       ↓
📋 View Conversation Details
```

## Key Components
- **Vector Store**: FAISS (CPU-based, local)
- **LLM Providers**: OpenAI, Claude, Ollama, HuggingFace
- **Database**: SQLite for user accounts & chat history
- **API Framework**: FastAPI with async support
- **Auth**: JWT-based authentication

## Data Flow Summary
1. **Ingestion**: Documents → Chunks → Embeddings → Vector DB
2. **Query**: Question → Search → LLM → Answer
3. **Persistence**: Conversations stored per user
4. **Admin**: View all user interactions & analytics
