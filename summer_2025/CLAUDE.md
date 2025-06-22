# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an ME344 educational project that implements a Retrieval Augmented Generation (RAG) system for slang translation using Llama 3.1 and vector databases. The system combines a Python-based RAG pipeline with a React frontend for interactive chat.

## Architecture

### Core Components

- **Python RAG Pipeline** (`rag.ipynb`): Jupyter notebook containing the complete RAG implementation
  - Data loading and chunking using LangChain
  - ChromaDB vector database integration
  - Ollama embedding functions with `nomic-embed-text` model
  - Document processing for 600k+ slang definitions

- **React Frontend** (`llm-rag-chat/`): Web interface for chat interaction
  - RAG component (`src/components/Rag.js`) handles query augmentation
  - Connects to ChromaDB (port 8000) and Ollama (port 11434)
  - Real-time embedding and context retrieval

- **Vector Database**: ChromaDB storing embedded slang definitions
  - Embedding function: `nomic-embed-text` via Ollama
  - Collection name: `llm_rag_collection`
  - Document chunking: 800 chars with 80 char overlap

### System Architecture Flow

1. **Data Pipeline**: CSV → LangChain loader → chunking → embedding → ChromaDB
2. **Query Processing**: User query → embedding → similarity search → context augmentation → LLM
3. **Response Generation**: Augmented query sent to Ollama LLM → response to frontend

## Development Commands

### Python Environment
```bash
# Activate virtual environment (required)
source ~/codes/python/python-venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start ChromaDB server
export CHROMA_SERVER_CORS_ALLOW_ORIGINS='["http://localhost:3000"]'
chroma run --host localhost --port 8000 --path ./chroma
```

### Ollama Model Management
```bash
# Run Llama 3.1 model (downloads if needed)
ollama run llama3.1

# Pull embedding model
ollama pull nomic-embed-text
```

### React Frontend
```bash
cd llm-rag-chat
npm install
npm start  # Starts on localhost:3000
```

### Jupyter Notebook
```bash
# Launch notebook server
jupyter-notebook --no-browser --notebook-dir=$PWD
```

## Key Configuration

- **Ollama API**: `http://localhost:11434`
- **ChromaDB**: `http://localhost:8000` 
- **React Dev Server**: `http://localhost:3000`
- **Embedding Model**: `nomic-embed-text`
- **LLM Model**: `llama3.1`
- **Collection**: `llm_rag_collection`

## Data Processing

The system processes Urban Dictionary slang data:
- Source: `data/cleaned_slang_data.csv` (628,985 entries)
- Format: word/definition pairs
- Chunking: RecursiveCharacterTextSplitter with 800/80 char settings
- Storage: ChromaDB with Ollama embeddings

## Important Notes

- Designed for GPU cluster deployment with SLURM
- Requires root access for Ollama installation
- Uses port forwarding for remote development
- RAG query augmentation happens in `Rag.js:40`
- System prompt engineering can be modified in `Rag.js`