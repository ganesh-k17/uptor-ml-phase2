# Agentic RAG Streamlit App

A Streamlit chatbot that answers questions from `company_policy.pdf` using:

- LangChain
- Ollama with the local `llama3.1` model
- Hugging Face embeddings
- FAISS vector search
- Streamlit chat interface

## Requirements

- Python 3.12 or compatible Python version
- The repository `.venv` virtual environment
- Ollama installed and available as the `ollama` command
- `company_policy.pdf` in this directory

## Setup

From this directory:

```bash
source ../../.venv/bin/activate
python -m pip install -r ../../requirements.txt
```

If the requirements file is not available, install the main packages directly:

```bash
python -m pip install streamlit langchain langchain-ollama \
    langchain-huggingface langchain-community langchain-text-splitters \
    faiss-cpu
```

## Start Ollama

Keep Ollama running in a separate terminal:

```bash
ollama serve
```

In another terminal, download the model used by the application:

```bash
ollama pull llama3.1
```

Verify the model if needed:

```bash
ollama list
ollama run llama3.1 "Reply with OK only."
```

## Run the Application

From this directory:

```bash
cd 007-Aug-2026/20
source ../../.venv/bin/activate
streamlit run streamlit_agentic_rag.py
```

Open the URL shown by Streamlit, usually:

```text
http://localhost:8501
```

Do not run the app with `python streamlit_agentic_rag.py`. Streamlit applications must be started with `streamlit run`.

## First Run

1. The app loads the Hugging Face embedding model.
2. If `faiss_index/` does not exist, it reads `company_policy.pdf`.
3. The PDF is split into text chunks.
4. The chunks are embedded and saved in the FAISS index.
5. The agent uses the index to answer policy questions.

Later runs reuse `faiss_index/`. If you replace or edit the PDF, delete the generated index and restart the app:

```bash
rm -rf faiss_index
```

## Example Questions

```text
What is the leave policy?
How many work-from-home days are allowed?
What are the expense reimbursement rules?
What is the IT security policy?
```

## Troubleshooting

### `model 'llama3.1' not found`

Download the model:

```bash
ollama pull llama3.1
```

### `Connection refused`

Start Ollama and leave it running:

```bash
ollama serve
```

### `NoSessionContext` or `missing ScriptRunContext`

The app was launched as a normal Python script. Use:

```bash
streamlit run streamlit_agentic_rag.py
```

### `langchain_ollama` cannot be imported

Install the integration in the active virtual environment:

```bash
python -m pip install langchain-ollama
```

## Generated Files

The application generates `faiss_index/`, which contains the FAISS database and should not be committed to Git. The policy PDF is the application input and is intentionally kept in the repository.
