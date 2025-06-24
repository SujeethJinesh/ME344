# ME344 RAG Project - Quick Reference

## 🚀 Quick Commands

### Start Everything
```bash
./run_part1.sh              # Basic RAG system
./run_part2.sh              # Deep Research Agent
```

### Start Individual Services
```bash
# ChromaDB
chroma run --host localhost --port 8000 --path ./chroma

# Ollama
ollama serve
ollama run llama3.1

# React Frontend
cd llm-rag-chat && npm start

# Jupyter
jupyter notebook
```

## 📍 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| React UI | http://localhost:3000 | Chat interface |
| ChromaDB | http://localhost:8000 | Vector database |
| Jupyter | http://localhost:8888 | RAG setup notebook |
| Ollama | http://localhost:11434 | LLM API |
| Research Agent | http://localhost:8001 | Part 2 only |
| MCP Server | http://localhost:8002 | Part 2 only |

## 🔧 Key Files

| File | Purpose |
|------|---------|
| `rag.ipynb` | Set up vector database with slang data |
| `llm-rag-chat/src/components/Rag.js` | RAG implementation |
| `llm-rag-chat/src/App.js` | Main chat interface |
| `deep_research_agent/main.py` | LangGraph research workflow |
| `mcp_server/main.py` | Web search tool server |

## 🛠️ Troubleshooting

### ChromaDB Issues
```bash
# Check if running
lsof -i :8000

# Kill and restart
lsof -ti :8000 | xargs kill -9
./run_part1.sh
```

### Jupyter Kernel
- Always select: **Kernel → Change Kernel → ME344 RAG (Python)**

### Port Forwarding (Remote)
```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 -L 8888:localhost:8888 -L 11434:localhost:11434 user@server
```

## 📊 Data Flow

### Part 1 - RAG Pipeline
1. User query → React UI
2. Query → Embedding (Ollama)
3. Embedding → Vector search (ChromaDB)
4. Context + Query → LLM (Ollama)
5. Response → React UI

### Part 2 - Research Agent
1. User query → React UI
2. Query → Research Agent (LangGraph)
3. Agent → Web search (MCP/Tavily)
4. Results → ChromaDB update
5. Enhanced RAG → Response

## 🔑 Environment Variables

```bash
# Required for Part 2
export TAVILY_API_KEY="your-api-key"

# Optional
export DOCUMENTS_TO_ADD=500  # Number of slang entries to index
```