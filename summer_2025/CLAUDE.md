# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an ME344 educational project that implements a Retrieval Augmented Generation (RAG) system for slang translation using Llama 3.1 and vector databases. The system has evolved to include multiple architectures: a basic RAG system, an advanced deep research agent with LangGraph, and an MCP (Model Context Protocol) server for web search capabilities.

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
  - Component structure: Header, Sidebar, ChatBox, InputBar, Message, Controls

- **Deep Research Agent** (`deep_research_agent/main.py`): Advanced research system

  - LangGraph-based multi-node workflow
  - FastAPI server (port 8001) with streaming responses
  - Multi-step research process: planning → search → processing → synthesis
  - Integrates with MCP server for web search capabilities

- **MCP Server** (`mcp_server/main.py`): Web search tool server

  - Tavily API integration for web search
  - FastMCP server (port 8002) providing search tools
  - JSON-RPC protocol for tool communication

- **Vector Database**: ChromaDB storing embedded slang definitions
  - Embedding function: `nomic-embed-text` via Ollama
  - Collection name: `llm_rag_collection`
  - Document chunking: 800 chars with 80 char overlap

### System Architecture Flow

#### Basic RAG System:

1. **Data Pipeline**: CSV → LangChain loader → chunking → embedding → ChromaDB
2. **Query Processing**: User query → embedding → similarity search → context augmentation → LLM
3. **Response Generation**: Augmented query sent to Ollama LLM → response to frontend

#### Deep Research Agent Flow:

1. **Planning**: Query analysis and search strategy generation
2. **Web Search**: MCP client calls Tavily API via MCP server
3. **Processing**: Content chunking and embedding
4. **Vector Store Update**: New knowledge added to ChromaDB
5. **RAG Retrieval**: Context retrieval from updated knowledge base
6. **Synthesis**: Final answer generation with citations

## Development Commands

### Automated Startup (Recommended)

```bash
# Start all services (Part 2 - Deep Research Agent)
./run_part2.sh
# This starts: MCP server (8002), Deep Research Agent (8001), ChromaDB (8000), Ollama, React frontend (3000)
```

### Manual Service Management

#### Python Environment

```bash
# Activate virtual environment (required)
source ~/codes/python/.venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start ChromaDB server
export CHROMA_SERVER_CORS_ALLOW_ORIGINS='["http://localhost:3000"]'
chroma run --host localhost --port 8000 --path ./chroma
```

#### MCP Server (Web Search)

```bash
# Requires TAVILY_API_KEY environment variable
export TAVILY_API_KEY="your_api_key"
cd mcp_server
python main.py  # Starts on port 8002
```

#### Deep Research Agent

```bash
cd deep_research_agent
uvicorn main:app --reload --port 8001
```

#### Ollama Model Management

```bash
# Start Ollama server
ollama serve

# Run Llama 3.1 model (downloads if needed)
ollama run llama3.1

# Pull embedding model
ollama pull nomic-embed-text
```

#### React Frontend

```bash
cd llm-rag-chat
npm install
npm start  # Starts on localhost:3000
```

#### Jupyter Notebook

```bash
# Launch notebook server
jupyter-notebook --no-browser --notebook-dir=$PWD
```

## Key Configuration

### Service Ports

- **React Dev Server**: `http://localhost:3000`
- **ChromaDB**: `http://localhost:8000`
- **Deep Research Agent**: `http://localhost:8001`
- **MCP Server**: `http://localhost:8002`
- **Ollama API**: `http://localhost:11434`

### Models and Collections

- **Embedding Model**: `nomic-embed-text`
- **LLM Model**: `llama3.1`
- **ChromaDB Collection**: `llm_rag_collection`

### Environment Variables

- `TAVILY_API_KEY`: Required for MCP web search functionality
- `REACT_APP_CHROMA_URL`: ChromaDB URL (default: http://localhost:8000)
- `REACT_APP_OLLAMA_URL`: Ollama API URL (default: http://localhost:11434)
- `REACT_APP_EMBEDDING_MODEL`: Embedding model name (default: nomic-embed-text)

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
- RAG query augmentation happens in `Rag.js:43-53`
- System prompt engineering can be modified in `Rag.js:63`
- Deep Research Agent uses LangGraph for multi-step workflows
- MCP protocol enables tool integration between components
- Startup script path issue: `run_part2.sh` references `./chat-gpt-clone` but directory is `./llm-rag-chat`

## Dependencies

### Python (requirements.txt)

- LangChain ecosystem (langchain, langchain-community, langchain-text-splitters)
- LangGraph for workflow orchestration
- FastAPI for web service
- ChromaDB for vector storage
- Ollama integrations
- Tavily for web search
- HTTPx for HTTP client operations

### React (package.json)

- React 18.3.1
- ChromaDB JavaScript client
- Workbox for service worker functionality

## Current Architecture Status

- **Basic RAG**: Fully functional with Jupyter notebook pipeline
- **React Frontend**: Complete with error handling and environment configuration
- **Deep Research Agent**: Functional but startup script has path error
- **MCP Server**: Functional web search integration via Tavily API
