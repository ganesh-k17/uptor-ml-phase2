# Harness Engineering Run Guide

This folder contains a local policy-assistant harness with:

- `create_holiday_db.py`: creates the local SQLite holiday database.
- `holiday_api.py`: exposes the holiday database through a REST API.
- `rag_engine.py`: shared RAG, re-ranking, tools, and conversation memory.
- `streamlit_full_harness.py`: Streamlit chat application using Ollama.
- `mcp_server_full.py`: MCP server exposing the same tools to MCP clients.
- `company_policy.pdf`: source document for policy questions.

## Part I: Quick Start

Use this short guide when the environment has already been set up.

### One-time preparation

```bash
cd /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2
source .venv/bin/activate
cd 008-Sep-2026/01-Tue/Harness_Engineering
python create_holiday_db.py
ollama pull llama3.1
```

### Run the Streamlit application

Open three terminals:

```text
Terminal 1: ollama serve
Terminal 2: uvicorn holiday_api:app --port 8000
Terminal 3: streamlit run streamlit_full_harness.py
```

Open the Streamlit URL, normally `http://localhost:8501`, and try:

```text
Is 2026-01-26 a public holiday in India?
```

### Run with Claude Desktop

1. Add `company-policy` to Claude Desktop's `claude_desktop_config.json` using the configuration shown in Part II.
2. Start the Holiday API in a terminal:

```bash
uvicorn holiday_api:app --port 8000
```

3. Quit and reopen Claude Desktop with `Cmd+Q`.
4. Open **Settings -> Developer** and confirm `company-policy` is connected.
5. Ask Claude:

```text
Use the holiday tool to check whether 2026-01-26 is a public holiday in India.
```

Claude Desktop starts the MCP server automatically. Do not also run `mcp dev` or `uv run ...` for this approach.

### Run MCP manually

For MCP Inspector or manual MCP testing, keep the Holiday API running and use a second terminal:

```bash
source /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/.venv/bin/activate
mcp dev mcp_server_full.py
```

Alternatively:

```bash
uv run --with mcp==2.1.1 mcp run mcp_server_full.py
```

The first MCP or Streamlit run may take a few minutes while the embedding and re-ranking models load.

---

## Part II: Detailed Guide

## 1. One-Time Setup

Run these commands from the project root or use the absolute paths shown below.

```bash
cd /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2
source .venv/bin/activate
cd 008-Sep-2026/01-Tue/Harness_Engineering
```

Install dependencies into the active virtual environment:

```bash
pip install -r /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/requirements.txt
pip install fastapi uvicorn langchain-experimental sentence-transformers mcp
```

Confirm the important tools are available:

```bash
python --version
python -c "import fastapi, streamlit, langchain, mcp; print('Dependencies OK')"
ollama --version
```

Download the local LLM used by the Streamlit app:

```bash
ollama pull llama3.1
```

The first RAG startup also downloads Hugging Face models. This can take several minutes the first time.

## 2. One-Time Data Initialization

Create or refresh the holiday database:

```bash
cd /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/008-Sep-2026/01-Tue/Harness_Engineering
python create_holiday_db.py
```

This creates `holidays.db`. The script drops and recreates the table, so run it again whenever the holiday list changes.

The following files are created or reused automatically:

- `holidays.db`: holiday records.
- `faiss_index_semantic/`: semantic FAISS index built from `company_policy.pdf`.
- `conversation_memory.db`: persistent conversation and MCP tool-call history.

Do not delete `company_policy.pdf` unless you plan to provide another policy document.

## 3. Approach A: Test the Holiday REST API Only

This tests the database and FastAPI layer without Ollama, Streamlit, RAG, or MCP.

### Terminal A1: start the API

```bash
cd /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/008-Sep-2026/01-Tue/Harness_Engineering
source /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/.venv/bin/activate
uvicorn holiday_api:app --reload --port 8000
```

Keep this terminal running.

### Terminal A2: test an India holiday

```bash
curl "http://127.0.0.1:8000/holidays?date=2026-01-26&country_code=IN"
```

Expected result:

```json
{"date":"2026-01-26","country_code":"IN","is_holiday":true,"name":"Republic Day","local_name":"Republic Day"}
```

Other useful endpoints:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/holidays/IN/2026
http://127.0.0.1:8000/holidays?date=2026-07-04&country_code=US
```

## 4. Approach B: Run the Streamlit Chat App

The Streamlit app needs both Ollama and the Holiday API. Start each long-running service in its own terminal.

### Terminal B1: start Ollama

```bash
ollama serve
```

If Ollama is already running, leave it as it is. Check the model exists:

```bash
ollama list
```

### Terminal B2: start the Holiday API

```bash
cd /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/008-Sep-2026/01-Tue/Harness_Engineering
source /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/.venv/bin/activate
uvicorn holiday_api:app --reload --port 8000
```

### Terminal B3: start Streamlit

```bash
cd /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/008-Sep-2026/01-Tue/Harness_Engineering
source /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/.venv/bin/activate
streamlit run streamlit_full_harness.py
```

Open the URL printed by Streamlit, normally:

```text
http://localhost:8501
```

### Example Streamlit questions

Holiday tool:

```text
Is 2026-01-26 a public holiday in India?
```

```text
Is 2026-07-04 a public holiday in the US?
```

Policy RAG tool:

```text
What is the company's work-from-home policy?
```

```text
What does the policy say about leave?
```

HR ticket tool, after the policy conversation:

```text
I confirm that you should create an HR ticket for my leave request. My employee ID is E1001, the request type is leave, and the details are leave from 2026-09-10 to 2026-09-12.
```

The Streamlit app uses `system_prompt=SYSTEM_PROMPT`. Do not change it back to `prompt=SYSTEM_PROMPT`; newer LangChain versions reject the old keyword.

## 5. Approach C: Run and Test the MCP Server Locally

The MCP server uses stdio. It does not provide its own chat UI or LLM. An MCP client must connect to it.

The MCP server loads the FAISS index and re-ranking model at startup. The first startup can take several minutes; later starts use the local files and model cache.

### Option C1: run with the installed MCP command

From `Harness_Engineering`:

```bash
source /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/.venv/bin/activate
mcp dev mcp_server_full.py
```

If `mcp` is not found, install it in the active environment:

```bash
pip install mcp
```

### Option C2: run with uv

This uses an isolated temporary environment for the requested MCP version:

```bash
cd /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/008-Sep-2026/01-Tue/Harness_Engineering
uv run --with mcp==2.1.1 mcp run mcp_server_full.py
```

Do not run this command manually while Claude Desktop is already configured to launch the same server. Choose either the MCP client or the manual test client.

### MCP tools exposed

- `search_policy_tool(query)`: searches `company_policy.pdf` using semantic retrieval and re-ranking.
- `check_public_holiday_tool(date, country_code)`: calls the local Holiday API on port 8000.
- `raise_hr_ticket_tool(employee_id, request_type, details)`: returns a mock HR ticket and should only be used after explicit confirmation.

The MCP server does not require Streamlit or Ollama. It does require the Holiday API if you call `check_public_holiday_tool`.

## 6. Approach D: Connect MCP to Claude Desktop on macOS

Claude Desktop launches `mcp_server_full.py` itself. Do not run `mcp dev` or `uv run ...` manually for this approach.

### 6.1 Configuration file

Open the Claude Desktop configuration file:

```bash
open -e "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
```

The `mcpServers` object must be at the top level of the JSON file, not inside `preferences`.

Add or merge this block at the top level:

```json
{
  "mcpServers": {
    "company-policy": {
      "command": "/Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/.venv/bin/python",
      "args": [
        "/Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/008-Sep-2026/01-Tue/Harness_Engineering/mcp_server_full.py"
      ]
    }
  }
}
```

If the file already has `preferences` or other top-level settings, keep them and add `mcpServers` beside them. Do not create a second JSON root object.

### 6.2 Start the dependency used by the holiday tool

In a separate terminal, keep the local Holiday API running:

```bash
cd /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/008-Sep-2026/01-Tue/Harness_Engineering
source /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/.venv/bin/activate
uvicorn holiday_api:app --port 8000
```

### 6.3 Restart Claude Desktop

1. Save `claude_desktop_config.json`.
2. Quit Claude Desktop completely with `Cmd+Q`.
3. Reopen Claude Desktop.
4. Open **Settings -> Developer** and find `company-policy`.
5. Look for a connected/running status and the available tools.

The tools should be named:

- `search_policy_tool`
- `check_public_holiday_tool`
- `raise_hr_ticket_tool`

You can verify the connection by checking the developer/server logs without sending a chat prompt. A successful startup contains text similar to:

```text
Server started and connected successfully
```

A failed startup usually contains:

```text
sqlite3.OperationalError: unable to open database file
Server disconnected
```

The shared RAG paths are resolved relative to `rag_engine.py`, so Claude Desktop can launch the server from a different working directory.

## 7. Recommended Terminal Layout

For Streamlit:

```text
Terminal 1: ollama serve
Terminal 2: uvicorn holiday_api:app --reload --port 8000
Terminal 3: streamlit run streamlit_full_harness.py
```

For Claude Desktop MCP:

```text
Terminal 1: uvicorn holiday_api:app --port 8000
Claude Desktop: launches mcp_server_full.py from claude_desktop_config.json
```

For MCP manual testing:

```text
Terminal 1: uvicorn holiday_api:app --port 8000
Terminal 2: mcp dev mcp_server_full.py
```

## 8. Troubleshooting

### `sqlite3.OperationalError: unable to open database file`

Make sure `conversation_memory.db` is writable and use the configured `.venv/bin/python`. The current `rag_engine.py` resolves its database path from its own directory, which avoids Claude Desktop working-directory problems.

### `Could not reach the local holiday API`

Check that the API is running on port 8000:

```bash
curl "http://127.0.0.1:8000/holidays?date=2026-01-26&country_code=IN"
```

### `No module named ...`

Activate the project environment and install dependencies there:

```bash
source /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/.venv/bin/activate
pip install -r /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/requirements.txt
pip install fastapi uvicorn langchain-experimental sentence-transformers mcp
```

### `mcp: command not found`

```bash
source /Users/ganesh/ganku/code/Uptor-AI-ML/Uptor-ML-Phase2/.venv/bin/activate
pip install mcp
```

Or use:

```bash
uv run --with mcp==2.1.1 mcp run mcp_server_full.py
```

### `ollama` errors in Streamlit

```bash
ollama serve
ollama pull llama3.1
ollama list
```

### Hugging Face warning

An unauthenticated Hugging Face warning is normally not fatal. The embedding and re-ranking models can still download, but setting `HF_TOKEN` can improve download rate limits.

### Model loading appears stuck

The first run downloads and loads two models:

- `sentence-transformers/all-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

Wait for the loading messages to finish. If the process exits, inspect the final traceback rather than only the last progress line.

## 9. Stop the Services

In each terminal running a foreground service, press:

```text
Ctrl+C
```

Quit Streamlit and Claude Desktop normally. The generated databases and FAISS index can remain for the next run.
