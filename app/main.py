from pathlib import Path
import re
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from app.config import settings
from app.db import (
    authenticate_user,
    create_user,
    get_chat_history_grouped_by_user,
    get_user_chat_history,
    init_db,
    save_chat,
)
from app.ingest import index_data_directory, index_files
from app.models import ChatRequest, ChatResponse, IngestResponse
from app.rag import RAGService
from app.utils import ensure_directory
from app.vector_store import get_vector_store_overview

app = FastAPI(title="RAG Chatbot", version="1.0.0")
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.app_secret_key,
    same_site="lax",
    https_only=False,
)
rag_service = RAGService()
init_db(settings.db_path, settings.admin_username, settings.admin_password)


@app.on_event("startup")
def startup() -> None:
    init_db(settings.db_path, settings.admin_username, settings.admin_password)


def _require_user(request: Request) -> dict:
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    return user


def _require_admin(request: Request) -> dict:
    user = _require_user(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/login", status_code=302)
    return RedirectResponse(url="/chat-ui", status_code=302)


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Login - RAG Chatbot</title>
  <style>
    :root { color-scheme: light; }
    * { box-sizing: border-box; }
    body {
      font-family: Inter, Segoe UI, Arial, sans-serif;
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at top left, #1f2937 0%, #0f172a 55%, #020617 100%);
      color: #e2e8f0;
      display: grid;
      place-items: center;
      padding: 20px;
    }
    .shell {
      width: 100%;
      max-width: 840px;
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 18px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
      padding: 24px;
      backdrop-filter: blur(6px);
    }
    h1 { margin: 0 0 6px; }
    .subtitle { margin: 0 0 16px; color: #94a3b8; }
    .card {
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 12px;
      padding: 16px;
    }
    input {
      width: 100%;
      padding: 11px 12px;
      margin: 6px 0;
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: #0b1220;
      color: #e2e8f0;
    }
    button {
      padding: 10px 14px;
      cursor: pointer;
      border: 0;
      border-radius: 10px;
      background: linear-gradient(90deg, #22d3ee 0%, #a78bfa 100%);
      color: #0b1020;
      font-weight: 600;
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .status { color: #fda4af; min-height: 20px; margin-top: 12px; }
    @media (max-width: 760px) { .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="shell">
    <h1>RAG Chatbot</h1>
    <p class="subtitle">Sign in to continue, or create a new account.</p>
    <div class="row">
      <div class="card">
        <h3>Login</h3>
        <input id="loginUsername" type="text" placeholder="Username" />
        <input id="loginPassword" type="password" placeholder="Password" />
        <button onclick="login()">Login</button>
      </div>
      <div class="card">
        <h3>Register</h3>
        <input id="registerUsername" type="text" placeholder="Username" />
        <input id="registerPassword" type="password" placeholder="Password" />
        <button onclick="registerUser()">Register</button>
      </div>
    </div>
    <div id="status" class="status"></div>
  </div>

  <script>
    async function login() {
      const username = document.getElementById("loginUsername").value.trim();
      const password = document.getElementById("loginPassword").value;
      const status = document.getElementById("status");
      if (!username || !password) {
        status.textContent = "Enter username and password.";
        return;
      }
      const form = new URLSearchParams();
      form.append("username", username);
      form.append("password", password);
      const res = await fetch("/login", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: form.toString()
      });
      if (!res.ok) {
        const data = await res.json();
        status.textContent = data.detail || "Login failed.";
        return;
      }
      window.location.href = "/chat-ui";
    }

    async function registerUser() {
      const username = document.getElementById("registerUsername").value.trim();
      const password = document.getElementById("registerPassword").value;
      const status = document.getElementById("status");
      if (!username || !password) {
        status.textContent = "Enter username and password.";
        return;
      }
      const form = new URLSearchParams();
      form.append("username", username);
      form.append("password", password);
      const res = await fetch("/register", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: form.toString()
      });
      const data = await res.json();
      if (!res.ok) {
        status.textContent = data.detail || "Registration failed.";
        return;
      }
      status.textContent = "Registered successfully. Now login.";
    }
  </script>
</body>
</html>
"""


@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    normalized = username.strip()
    if len(normalized) < 3 or len(password) < 4:
        raise HTTPException(status_code=400, detail="Username/password too short")
    created = create_user(settings.db_path, normalized, password)
    if not created:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"message": "User created"}


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(settings.db_path, username.strip(), password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    request.session["user"] = user
    return {"message": "Logged in", "user": {"username": user["username"], "is_admin": user["is_admin"]}}


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui(request: Request):
    user = _require_user(request)
    admin_button = (
        '<a href="/admin-ui"><button style="margin-left:8px;">Admin View</button></a>'
        if user.get("is_admin")
        else ""
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Chatbot</title>
  <style>
    :root {{ color-scheme: light; }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: Inter, Segoe UI, Arial, sans-serif;
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at top left, #0b1220 0%, #111827 45%, #030712 100%);
      color: #e5e7eb;
      padding: 20px;
    }}
    .shell {{ max-width: 1320px; margin: 0 auto; }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(17, 24, 39, 0.75);
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 14px;
      backdrop-filter: blur(6px);
    }}
    .header h1 {{ margin: 0; }}
    .muted {{ color: #93c5fd; font-size: 13px; }}
    .actions {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    .main-layout {{
      display: grid;
      grid-template-columns: 2.2fr 1fr;
      gap: 14px;
      align-items: start;
    }}
    .left-stack {{ display: grid; gap: 14px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .card {{
      background: rgba(17, 24, 39, 0.7);
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 12px 24px rgba(0, 0, 0, 0.22);
    }}
    .card h3 {{ margin-top: 0; }}
    input[type="text"], input[type="file"] {{
      width: 100%;
      padding: 11px 12px;
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: #0b1220;
      color: #e5e7eb;
      margin-bottom: 8px;
    }}
    button {{
      padding: 10px 14px;
      border: 0;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
      background: linear-gradient(90deg, #22d3ee 0%, #a78bfa 100%);
      color: #0b1020;
    }}
    .status {{
      white-space: pre-wrap;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 10px;
      padding: 10px;
      min-height: 44px;
    }}
    .answer {{ border-left: 4px solid #22d3ee; }}
    .history-list {{ display: grid; gap: 10px; }}
    .history-item {{
      background: rgba(2, 6, 23, 0.75);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 10px;
      padding: 10px;
    }}
    .q {{ color: #93c5fd; margin-bottom: 4px; }}
    .a {{ color: #e2e8f0; }}
    .right-stack {{ display: grid; gap: 14px; position: sticky; top: 14px; }}
    .process-title {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .step-list {{ display: grid; gap: 8px; }}
    .step {{
      display: flex;
      align-items: flex-start;
      gap: 10px;
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 10px;
      background: rgba(2, 6, 23, 0.65);
      padding: 10px;
    }}
    .dot {{
      width: 11px;
      height: 11px;
      border-radius: 50%;
      margin-top: 4px;
      background: #64748b;
      box-shadow: 0 0 0 4px rgba(100, 116, 139, 0.2);
    }}
    .step.pending .dot {{ background: #64748b; box-shadow: 0 0 0 4px rgba(100, 116, 139, 0.2); }}
    .step.running .dot {{ background: #f59e0b; box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.25); }}
    .step.done .dot {{ background: #22c55e; box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.2); }}
    .step.error .dot {{ background: #ef4444; box-shadow: 0 0 0 4px rgba(239, 68, 68, 0.2); }}
    .step-label {{ font-weight: 600; }}
    .step-detail {{ font-size: 12px; color: #94a3b8; margin-top: 2px; }}
    .kv {{ display: grid; grid-template-columns: 130px 1fr; gap: 6px; margin-bottom: 6px; font-size: 13px; }}
    .kv .k {{ color: #93c5fd; }}
    .source-list {{ max-height: 220px; overflow: auto; padding-left: 18px; margin: 8px 0 0; }}
    .source-list li {{ margin: 4px 0; color: #dbeafe; font-size: 12px; word-break: break-all; }}
    .trace-list {{
      margin: 0;
      padding: 0;
      list-style: none;
      max-height: 280px;
      overflow: auto;
      display: grid;
      gap: 6px;
    }}
    .trace-line {{
      background: rgba(2, 6, 23, 0.65);
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 12px;
      color: #dbeafe;
    }}
    .flow-list {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 6px;
      max-height: 280px;
      overflow: auto;
    }}
    .flow-item {{
      background: rgba(2, 6, 23, 0.65);
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 12px;
      color: #dbeafe;
    }}
    .flow-arrow {{
      text-align: center;
      color: #94a3b8;
      font-size: 12px;
      line-height: 1;
      margin-top: -2px;
    }}
    .rag-flow {{ display: grid; gap: 8px; }}
    .rag-chip {{
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 8px;
      background: rgba(2, 6, 23, 0.65);
      padding: 8px 10px;
      font-size: 12px;
      color: #dbeafe;
    }}
    .rag-arrow {{
      text-align: center;
      color: #94a3b8;
      font-size: 12px;
      margin: -2px 0;
    }}
    .toggle-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 6px 0 10px;
      color: #cbd5e1;
      font-size: 13px;
    }}
    @media (max-width: 1120px) {{
      .main-layout {{ grid-template-columns: 1fr; }}
      .right-stack {{ position: static; }}
    }}
    @media (max-width: 920px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div>
        <h1>RAG Chatbot</h1>
        <div class="muted">Ollama + FAISS</div>
      </div>
      <div class="actions">
        <strong>User:</strong> {user["username"]}
        {admin_button}
        <a href="/logout"><button>Logout</button></a>
      </div>
    </div>

    <div class="main-layout">
      <div class="left-stack">
        <div class="grid">
          <div class="card">
            <h3>Ingest data/ directory</h3>
            <button onclick="ingestDir()">Ingest Directory</button>
            <div id="ingestDirStatus" class="status"></div>
          </div>
          <div class="card">
            <h3>Upload and ingest file</h3>
            <input id="fileInput" type="file" />
            <button onclick="uploadFile()">Upload & Ingest</button>
            <div id="uploadStatus" class="status"></div>
          </div>
        </div>

        <div class="card">
          <h3>Ask question</h3>
          <input id="question" type="text" placeholder="Type your question..." />
          <label class="toggle-row">
            <input id="useDocuments" type="checkbox" />
            Use uploaded files for this question (RAG)
          </label>
          <button onclick="ask()">Send</button>
          <h4>Answer</h4>
          <div id="answer" class="status answer"></div>
          <h4>Sources</h4>
          <div id="sources" class="status"></div>
        </div>

        <div class="card">
          <h3>Your chat history</h3>
          <button onclick="loadHistory()">Refresh History</button>
          <div id="history" class="history-list"></div>
        </div>
      </div>

      <div class="right-stack">
        <div class="card">
          <div class="process-title">
            <h3 style="margin:0;">Backend Process</h3>
            <span class="muted" id="processMode">Idle</span>
          </div>
          <div id="processSteps" class="step-list"></div>
        </div>
        <div class="card">
          <div class="process-title">
            <h3 style="margin:0;">Vector Database</h3>
            <button onclick="loadVectorDb()">Refresh</button>
          </div>
          <div id="vectorDbStatus" class="status">Loading...</div>
          <ul id="vectorDbSources" class="source-list"></ul>
        </div>
        <div class="card">
          <div class="process-title">
            <h3 style="margin:0;">Chat Debug Trace</h3>
          </div>
          <ul id="chatDebugTrace" class="trace-list">
            <li class="trace-line">Waiting for next chat request...</li>
          </ul>
        </div>
        <div class="card">
          <div class="process-title">
            <h3 style="margin:0;">Message Execution Flow</h3>
          </div>
          <ul id="executionFlow" class="flow-list">
            <li class="flow-item">Type a message to see full flow.</li>
          </ul>
        </div>
        <div class="card">
          <div class="process-title">
            <h3 style="margin:0;">RAG Response Flow</h3>
          </div>
          <div id="ragFlowSummary" class="status">Run a chat to see RAG path.</div>
          <div id="ragFlowSteps" class="rag-flow"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const PROCESS_TEMPLATES = {{
      idle: [
        {{ id: "ready", label: "System ready", detail: "Waiting for action" }}
      ],
      ingest: [
        {{ id: "ingest-req", label: "Ingest request received", detail: "Sending /ingest-directory" }},
        {{ id: "ingest-read", label: "Reading files", detail: "Loading documents from data/" }},
        {{ id: "ingest-embed", label: "Creating embeddings", detail: "Running Ollama embedding model" }},
        {{ id: "ingest-index", label: "Updating vector index", detail: "Persisting FAISS index" }}
      ],
      upload: [
        {{ id: "upload-req", label: "Upload request received", detail: "Posting file to backend" }},
        {{ id: "upload-parse", label: "Parsing document", detail: "Loading uploaded file" }},
        {{ id: "upload-embed", label: "Creating embeddings", detail: "Running Ollama embedding model" }},
        {{ id: "upload-index", label: "Updating vector index", detail: "Persisting FAISS index" }}
      ],
      chat: [
        {{ id: "chat-req", label: "Chat request received", detail: "Posting question to /chat" }},
        {{ id: "chat-retrieve", label: "Retrieving context", detail: "Similarity search in FAISS" }},
        {{ id: "chat-generate", label: "Generating answer", detail: "Running Ollama chat model" }},
        {{ id: "chat-save", label: "Saving history", detail: "Writing response to SQLite" }},
        {{ id: "chat-refresh", label: "Refreshing UI history", detail: "Fetching latest /history" }}
      ],
      history: [
        {{ id: "history-req", label: "History request received", detail: "Fetching /history" }},
        {{ id: "history-render", label: "Rendering history", detail: "Updating history panel" }}
      ]
    }};

    function renderProcess(mode, initialState = "pending") {{
      document.getElementById("processMode").textContent = mode.charAt(0).toUpperCase() + mode.slice(1);
      const steps = PROCESS_TEMPLATES[mode] || PROCESS_TEMPLATES.idle;
      const container = document.getElementById("processSteps");
      container.innerHTML = steps.map((step) => `
        <div id="step-${{step.id}}" class="step ${{initialState}}">
          <div class="dot"></div>
          <div>
            <div class="step-label">${{step.label}}</div>
            <div class="step-detail">${{step.detail}}</div>
          </div>
        </div>
      `).join("");
      if (mode === "idle") {{
        setStepState("ready", "done", "UI ready for next request");
      }}
    }}

    function setStepState(id, state, detail = "") {{
      const node = document.getElementById(`step-${{id}}`);
      if (!node) return;
      node.className = `step ${{state}}`;
      if (detail) {{
        const detailNode = node.querySelector(".step-detail");
        if (detailNode) detailNode.textContent = detail;
      }}
    }}

    function setOperationError(stepId, message) {{
      setStepState(stepId, "error", message || "Operation failed");
    }}

    function renderDebugTrace(lines) {{
      const out = document.getElementById("chatDebugTrace");
      const rows = (lines || []).filter(Boolean);
      if (!rows.length) {{
        out.innerHTML = '<li class="trace-line">No debug lines available.</li>';
        return;
      }}
      out.innerHTML = rows.map((line) => `<li class="trace-line">${{line}}</li>`).join("");
    }}

    function renderExecutionFlow(lines) {{
      const out = document.getElementById("executionFlow");
      const rows = (lines || []).filter(Boolean);
      if (!rows.length) {{
        out.innerHTML = '<li class="flow-item">No execution flow available.</li>';
        return;
      }}
      out.innerHTML = rows.map((line, idx) => `
        <li class="flow-item">${{line}}</li>
        ${{idx < rows.length - 1 ? '<li class="flow-arrow">↓</li>' : ""}}
      `).join("");
    }}

    function renderRagFlow(flow) {{
      const summary = document.getElementById("ragFlowSummary");
      const stepsOut = document.getElementById("ragFlowSteps");
      if (!flow || !Object.keys(flow).length) {{
        summary.textContent = "No RAG flow available for this response.";
        stepsOut.innerHTML = "";
        return;
      }}

      summary.innerHTML = `
        <div class="kv"><span class="k">Mode</span><span>${{flow.mode || "unknown"}}</span></div>
        <div class="kv"><span class="k">Doc mode</span><span>${{flow.used_documents ? "ON" : "OFF"}}</span></div>
        <div class="kv"><span class="k">Index found</span><span>${{flow.vector_index_found ? "Yes" : "No"}}</span></div>
        <div class="kv"><span class="k">Retrieved docs</span><span>${{flow.retrieved_docs ?? 0}}</span></div>
        <div class="kv"><span class="k">Context chars</span><span>${{flow.context_chars ?? 0}}</span></div>
      `;

      const steps = flow.steps || [];
      if (!steps.length) {{
        stepsOut.innerHTML = "";
        return;
      }}
      stepsOut.innerHTML = steps.map((step, idx) => `
        <div class="rag-chip">${{idx + 1}}. ${{step}}</div>
        ${{idx < steps.length - 1 ? '<div class="rag-arrow">↓</div>' : ""}}
      `).join("");
    }}

    async function ingestDir() {{
      renderProcess("ingest");
      const out = document.getElementById("ingestDirStatus");
      out.textContent = "Indexing...";
      setStepState("ingest-req", "running");
      const res = await fetch("/ingest-directory", {{ method: "POST" }});
      const data = await res.json();
      if (!res.ok) {{
        out.textContent = data.detail || "Request failed.";
        setOperationError("ingest-req", data.detail || "Failed to send request");
        return;
      }}
      setStepState("ingest-req", "done");
      setStepState("ingest-read", "done", `Indexed files: ${{data.indexed_files ?? 0}}`);
      setStepState("ingest-embed", "done", `Indexed chunks: ${{data.indexed_chunks ?? 0}}`);
      setStepState("ingest-index", "done", "FAISS index updated");
      out.textContent = JSON.stringify(data, null, 2);
      await loadVectorDb();
    }}

    async function uploadFile() {{
      const fileInput = document.getElementById("fileInput");
      const out = document.getElementById("uploadStatus");
      if (!fileInput.files.length) {{
        out.textContent = "Select a file first.";
        renderProcess("upload");
        setOperationError("upload-req", "No file selected");
        return;
      }}

      renderProcess("upload");
      out.textContent = "Uploading...";
      setStepState("upload-req", "running");
      const form = new FormData();
      form.append("file", fileInput.files[0]);
      const res = await fetch("/ingest-file", {{ method: "POST", body: form }});
      const data = await res.json();
      if (!res.ok) {{
        out.textContent = data.detail || "Request failed.";
        setOperationError("upload-req", data.detail || "Upload failed");
        return;
      }}
      setStepState("upload-req", "done", "File uploaded successfully");
      setStepState("upload-parse", "done", data.metadata?.original_filename || "Document parsed");
      setStepState("upload-embed", "done", `Indexed chunks: ${{data.indexed_chunks ?? 0}}`);
      setStepState("upload-index", "done", "FAISS index updated");
      out.textContent = JSON.stringify(data, null, 2);
      await loadVectorDb();
    }}

    async function ask() {{
      const q = document.getElementById("question").value.trim();
      const answer = document.getElementById("answer");
      const sources = document.getElementById("sources");
      const useDocumentsToggle = document.getElementById("useDocuments");
      if (!q) {{
        answer.textContent = "Enter a question first.";
        renderProcess("chat");
        setOperationError("chat-req", "Question is empty");
        return;
      }}
      const useByText = /\\buse (the )?(file|files|document|documents|docs)\\b/i.test(q);
      const useDocuments = Boolean(useDocumentsToggle.checked || useByText);
      const payload = {{ question: q, use_documents: useDocuments }};

      renderProcess("chat");
      renderDebugTrace([
        "1. UI: Question submitted",
        `2. UI: Document mode = ${{useDocuments ? "ON" : "OFF"}}`,
        "3. UI: Sending POST /chat",
        "4. UI: Waiting for backend response"
      ]);
      renderExecutionFlow([
        "1. User typed question in UI",
        "2. Frontend prepared /chat payload",
        `3. Payload argument: ${{JSON.stringify(payload)}}`,
        "4. Request sent to FastAPI /chat endpoint",
        "5. Waiting for backend pipeline..."
      ]);
      answer.textContent = "Thinking...";
      sources.textContent = "";
      setStepState("chat-req", "running");
      const res = await fetch("/chat", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload)
      }});
      const data = await res.json();
      if (!res.ok) {{
        answer.textContent = data.detail || "Request failed.";
        setOperationError("chat-req", data.detail || "Chat request failed");
        renderRagFlow({{}});
        renderExecutionFlow([
          "1. User typed question in UI",
          "2. Frontend sent /chat payload",
          `3. Backend returned error: ${{data.detail || "Request failed"}}`
        ]);
        renderDebugTrace([
          "1. UI: Question submitted",
          "2. UI: POST /chat failed",
          `3. Error: ${{data.detail || "Request failed"}}`
        ]);
        return;
      }}
      setStepState("chat-req", "done");
      setStepState("chat-retrieve", "done", `Sources found: ${{(data.sources || []).length}}`);
      setStepState("chat-generate", "done");
      setStepState("chat-save", "done");
      answer.textContent = data.answer || "";
      sources.textContent = JSON.stringify(data.sources || [], null, 2);
      renderRagFlow(data.rag_process || {{}});
      renderExecutionFlow(data.execution_flow || []);
      renderDebugTrace(data.debug_trace || []);
      await loadHistory(false);
      setStepState("chat-refresh", "done");
    }}

    async function loadHistory(showProcess = true) {{
      if (showProcess) {{
        renderProcess("history");
      }}
      const out = document.getElementById("history");
      if (showProcess) {{
        setStepState("history-req", "running");
      }} else {{
        setStepState("chat-refresh", "running");
      }}
      const res = await fetch("/history");
      const data = await res.json();
      if (!res.ok) {{
        out.innerHTML = `<div class="status">${{data.detail || "Failed to load history."}}</div>`;
        if (showProcess) {{
          setOperationError("history-req", data.detail || "History request failed");
        }} else {{
          setOperationError("chat-refresh", data.detail || "History refresh failed");
        }}
        return;
      }}
      if (showProcess) {{
        setStepState("history-req", "done");
      }}
      const rows = data.history || [];
      if (!rows.length) {{
        out.innerHTML = '<div class="status">No chat history yet.</div>';
        if (showProcess) {{
          setStepState("history-render", "done", "No rows to render");
        }}
        return;
      }}
      out.innerHTML = rows.map((item) => `
        <div class="history-item">
          <div class="q"><strong>Q:</strong> ${{item.question}}</div>
          <div class="a"><strong>A:</strong> ${{item.answer}}</div>
          <div class="muted">${{item.created_at}}</div>
        </div>
      `).join("");
      if (showProcess) {{
        setStepState("history-render", "done", `Rendered ${{rows.length}} messages`);
      }}
    }}

    async function loadVectorDb() {{
      const status = document.getElementById("vectorDbStatus");
      const sources = document.getElementById("vectorDbSources");
      status.textContent = "Loading vector database...";
      sources.innerHTML = "";

      const res = await fetch("/vector-db");
      const data = await res.json();
      if (!res.ok) {{
        status.textContent = data.detail || "Failed to load vector database info.";
        return;
      }}

      if (!data.exists) {{
        status.textContent = `Not indexed yet. Path: ${{data.path}}`;
        return;
      }}

      const files = (data.files || []).map((f) => `${{f.name}} (${{f.size_bytes}} bytes)`).join(", ");
      status.innerHTML = `
        <div class="kv"><span class="k">Path</span><span>${{data.path}}</span></div>
        <div class="kv"><span class="k">Vectors</span><span>${{data.total_vectors}}</span></div>
        <div class="kv"><span class="k">Chunks</span><span>${{data.total_chunks}}</span></div>
        <div class="kv"><span class="k">Sources</span><span>${{data.source_count}}</span></div>
        <div class="kv"><span class="k">Files</span><span>${{files || "N/A"}}</span></div>
      `;
      const sourceRows = data.sources || [];
      if (!sourceRows.length) {{
        sources.innerHTML = "<li>No source metadata found.</li>";
        return;
      }}
      sources.innerHTML = sourceRows.map((s) => `<li>${{s}}</li>`).join("");
    }}

    renderProcess("idle");
    renderRagFlow({{}});
    renderExecutionFlow([]);
    loadHistory(false);
    loadVectorDb();
  </script>
</body>
</html>
"""


@app.get("/admin-ui", response_class=HTMLResponse)
def admin_ui(request: Request):
    _require_admin(request)
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Admin - RAG Chatbot</title>
  <style>
    body {
      font-family: Inter, Segoe UI, Arial, sans-serif;
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at top left, #0b1220 0%, #111827 45%, #030712 100%);
      color: #e5e7eb;
      padding: 20px;
    }
    .shell { max-width: 1050px; margin: 0 auto; }
    .card {
      background: rgba(17, 24, 39, 0.75);
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 14px;
      padding: 16px;
      margin: 12px 0;
    }
    .status { white-space: pre-wrap; background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 10px; }
    .user-block {
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 10px;
      padding: 12px;
      margin: 10px 0;
      background: rgba(2, 6, 23, 0.7);
    }
    .chat-item { border-top: 1px dashed rgba(148, 163, 184, 0.35); padding-top: 8px; margin-top: 8px; }
    .meta { color: #93c5fd; font-size: 12px; }
    button {
      padding: 10px 14px;
      margin-top: 8px;
      cursor: pointer;
      border: 0;
      border-radius: 10px;
      background: linear-gradient(90deg, #22d3ee 0%, #a78bfa 100%);
      color: #0b1020;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="shell">
    <h1>Admin: Chat History by User</h1>
    <a href="/chat-ui"><button>Back to Chat</button></a>
    <a href="/logout"><button>Logout</button></a>
    <div class="card">
      <button onclick="loadAll()">Refresh All History</button>
      <div id="allHistory"></div>
    </div>
  </div>
  <script>
    async function loadAll() {
      const out = document.getElementById("allHistory");
      const res = await fetch("/admin/history");
      const data = await res.json();
      if (!res.ok) {
        out.textContent = data.detail || "Failed to load.";
        return;
      }
      const grouped = data.grouped_history || {};
      const users = Object.keys(grouped);
      if (!users.length) {
        out.innerHTML = '<div class="status">No chat history found.</div>';
        return;
      }
      out.innerHTML = users.map((username) => {
        const chats = grouped[username] || [];
        const items = chats.map((chat) => `
          <div class="chat-item">
            <div class="meta"><strong>Time:</strong> ${chat.created_at}</div>
            <div><strong>Q:</strong> ${chat.question}</div>
            <div><strong>A:</strong> ${chat.answer}</div>
          </div>
        `).join("");
        return `
          <div class="user-block">
            <h3>${username} (${chats.length})</h3>
            ${items}
          </div>
        `;
      }).join("");
    }
    loadAll();
  </script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/me")
def me(request: Request):
    return {"user": _require_user(request)}


@app.get("/history")
def history(request: Request):
    user = _require_user(request)
    return {"history": get_user_chat_history(settings.db_path, user["id"])}


@app.get("/vector-db")
def vector_db_overview(request: Request):
    _require_user(request)
    return get_vector_store_overview(settings.vector_store_path)


@app.get("/admin/history")
def admin_history(request: Request):
    _require_admin(request)
    return {"grouped_history": get_chat_history_grouped_by_user(settings.db_path)}


@app.post("/ingest-directory", response_model=IngestResponse)
def ingest_directory(request: Request):
    _require_user(request)
    try:
        indexed_files, indexed_chunks, files = index_data_directory()
        return IngestResponse(
            indexed_files=indexed_files,
            indexed_chunks=indexed_chunks,
            files=files,
            metadata={"directory": str(settings.documents_dir)},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest-file", response_model=IngestResponse)
def ingest_file(request: Request, file: UploadFile = File(...)):
    _require_user(request)
    file_suffix = Path(file.filename).suffix.lower()
    if file_suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=400, detail="Only .txt, .md, and .pdf files are supported")

    uploads_dir = settings.documents_dir / "uploads"
    ensure_directory(uploads_dir)
    temp_path = uploads_dir / f"{uuid4().hex}{file_suffix}"

    with temp_path.open("wb") as f:
        f.write(file.file.read())

    try:
        indexed_files, indexed_chunks, files = index_files([temp_path])
        return IngestResponse(
            indexed_files=indexed_files,
            indexed_chunks=indexed_chunks,
            files=files,
            metadata={"original_filename": file.filename},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest):
    user = _require_user(request)
    try:
        execution_flow: list[str] = []
        execution_flow.append("1. /chat endpoint received request")
        execution_flow.append(
            f'2. Parsed argument question="{payload.question}"'
        )
        execution_flow.append(
            f"3. Parsed argument use_documents={payload.use_documents}"
        )
        text_trigger = bool(
            re.search(r"\buse (the )?(file|files|document|documents|docs)\b", payload.question, re.I)
        )
        execution_flow.append(
            f"4. Text trigger detection result={text_trigger}"
        )
        use_documents = payload.use_documents or text_trigger
        execution_flow.append(
            f"5. Resolved use_documents={use_documents}"
        )
        answer, sources, debug_trace, rag_process = rag_service.answer(
            payload.question, use_documents=use_documents
        )
        execution_flow.append(
            f'6. RAG service completed with mode="{rag_process.get("mode", "unknown")}" and {len(sources)} sources'
        )
        debug_trace.append("10. Saving chat entry to SQLite history")
        serialized_sources = [source.model_dump() for source in sources]
        save_chat(
            settings.db_path,
            user_id=user["id"],
            question=payload.question,
            answer=answer,
            sources=serialized_sources,
        )
        execution_flow.append("7. Chat response persisted to SQLite history")
        debug_trace.append("11. Chat response ready and returned to UI")
        execution_flow.append("8. Response returned to frontend")
        return ChatResponse(
            answer=answer,
            sources=sources,
            debug_trace=debug_trace,
            rag_process=rag_process,
            execution_flow=execution_flow,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
