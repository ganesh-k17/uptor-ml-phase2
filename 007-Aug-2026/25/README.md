# Semantic Re-ranking RAG Agent

A Streamlit chatbot that answers questions from `company_policy.pdf` using:

- LangChain and an Ollama local language model
- Hugging Face embeddings
- FAISS vector search
- Semantic chunking
- A cross-encoder re-ranker

## Requirements

- Python 3.12 or compatible Python version
- A Python virtual environment named `.venv` at the repository root
- Ollama installed and available as the `ollama` command
- The `company_policy.pdf` file in this directory

## Setup

From the repository root:

```bash
source .venv/bin/activate

python -m pip install streamlit langchain langchain-ollama \
    langchain-huggingface langchain-community langchain-experimental \
    sentence-transformers faiss-cpu
```

If your macOS Python environment reports a certificate error while installing, use:

```bash
python -m pip install streamlit langchain langchain-ollama \
    langchain-huggingface langchain-community langchain-experimental \
    sentence-transformers faiss-cpu \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

## Start Ollama

Keep Ollama running in a separate terminal:

```bash
ollama serve
```

In another terminal, download the model used by the program:

```bash
ollama pull llama3.1
```

You can verify the model with:

```bash
ollama list
ollama run llama3.1 "Reply with OK only."
```

## Run the Streamlit App

From the `25` directory:

```bash
cd 007-Aug-2026/25
source ../../.venv/bin/activate
streamlit run streamlit_agentic_rag_semantic_rerank.py
```

Open the URL shown by Streamlit, usually:

```text
http://localhost:8501
```

Do not run this application with `python streamlit_agentic_rag_semantic_rerank.py`. Streamlit applications must be started with `streamlit run`.

## First Run

1. The app loads the embedding model.
2. If `faiss_index_semantic/` does not exist, it reads `company_policy.pdf`.
3. The PDF is split using semantic chunking.
4. Embeddings are stored in a new FAISS index.
5. The app uses FAISS retrieval followed by cross-encoder re-ranking.

Later runs reuse the existing `faiss_index_semantic/` folder. If you replace or edit the PDF, delete that folder and restart the app so the index is rebuilt:

```bash
rm -rf faiss_index_semantic
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
streamlit run streamlit_agentic_rag_semantic_rerank.py
```

### `create_agent() got an unexpected keyword argument 'prompt'`

The current LangChain API uses `system_prompt=`, not `prompt=`. The included program already uses the current API.

## Generated Files

The program generates `faiss_index_semantic/`, which contains the FAISS index and should not be committed to Git. The PDF input and learning images are kept in the repository intentionally.
