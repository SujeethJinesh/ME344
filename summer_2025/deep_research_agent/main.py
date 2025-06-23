#!/usr/bin/env python3
"""
ME344 Deep Research Agent
LangGraph-powered multi-step research system with MCP integration

This agent orchestrates complex research workflows by combining web search,
document processing, vector storage, and RAG-based synthesis.
"""

import os
import sys
import json
import logging
import httpx
import uuid
from typing import TypedDict, List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import StateGraph, END

# ===================================================================
# CONFIGURATION AND LOGGING
# ===================================================================

# Configure logging with consistent format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8001
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVAL_K = 3
DEFAULT_TIMEOUT = 60
MAX_QUERY_LENGTH = 1000

# Environment configuration with validation
REQUIRED_ENV_VARS = {
    "MCP_SERVER_URL": "http://localhost:8002/mcp",
    "CHROMA_PERSIST_DIR": "./chroma",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "EMBEDDING_MODEL": "nomic-embed-text",
    "LLM_MODEL": "llama3.1"
}

# ===================================================================
# ENVIRONMENT VALIDATION
# ===================================================================

def validate_environment() -> tuple[bool, list[str]]:
    """
    Validate environment variables and system requirements.
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        errors.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required packages
    required_packages = [
        ("fastapi", "FastAPI web framework"),
        ("langchain", "LangChain framework"), 
        ("chromadb", "ChromaDB vector database"),
        ("httpx", "HTTP client"),
        ("langgraph", "LangGraph workflow orchestration")
    ]
    
    for package, description in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            errors.append(f"Missing package: {package} ({description})")
    
    # Check optional environment variables
    if not os.getenv("TAVILY_API_KEY"):
        warnings.append("TAVILY_API_KEY not set - web search may fail")
        warnings.append("  Solution: export TAVILY_API_KEY='your_api_key'")
    
    # Set default environment variables
    for var, default in REQUIRED_ENV_VARS.items():
        if not os.getenv(var):
            os.environ[var] = default
            logger.info(f"Set default {var}={default}")
    
    # Log warnings
    if warnings:
        logger.warning("Environment warnings:")
        for warning in warnings:
            logger.warning(f"  {warning}")
    
    return len(errors) == 0, errors

def print_startup_info() -> None:
    """Print comprehensive startup information."""
    print("=" * 70)
    print("🧠 ME344 Deep Research Agent")
    print("   LangGraph Multi-Step Research System")
    print("=" * 70)
    print()
    
    print("🔧 Configuration:")
    print(f"   Server: {SERVER_HOST}:{SERVER_PORT}")
    print(f"   MCP Server: {os.getenv('MCP_SERVER_URL')}")
    print(f"   ChromaDB: {os.getenv('CHROMA_PERSIST_DIR')}")
    print(f"   Ollama: {os.getenv('OLLAMA_BASE_URL')}")
    print(f"   LLM Model: {os.getenv('LLM_MODEL')}")
    print(f"   Embedding Model: {os.getenv('EMBEDDING_MODEL')}")
    print(f"   Tavily API: {'✅ Set' if os.getenv('TAVILY_API_KEY') else '❌ Missing'}")
    print()

# ===================================================================
# GLOBAL STATE AND INITIALIZATION
# ===================================================================

# Global components
llm: Optional[ChatOllama] = None
embedding_function: Optional[OllamaEmbeddings] = None
vectorstore: Optional[Chroma] = None
retriever = None
research_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("🚀 Initializing Deep Research Agent...")
    
    if not initialize_components():
        logger.error("❌ Failed to initialize components")
        raise RuntimeError("Component initialization failed")
    
    logger.info("✅ Deep Research Agent initialized successfully")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Deep Research Agent...")

def initialize_components() -> bool:
    """
    Initialize all required components with error handling.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global llm, embedding_function, vectorstore, retriever, research_agent
    
    try:
        # Initialize LLM
        logger.info("🔧 Initializing LLM...")
        llm = ChatOllama(
            model=os.getenv('LLM_MODEL'),
            base_url=os.getenv('OLLAMA_BASE_URL'),
            temperature=0
        )
        
        # Initialize embeddings
        logger.info("🔧 Initializing embeddings...")
        embedding_function = OllamaEmbeddings(
            model=os.getenv('EMBEDDING_MODEL'),
            base_url=os.getenv('OLLAMA_BASE_URL')
        )
        
        # Initialize vector store
        logger.info("🔧 Initializing vector store...")
        vectorstore = Chroma(
            collection_name="llm_rag_collection",
            embedding_function=embedding_function,
            persist_directory=os.getenv('CHROMA_PERSIST_DIR'),
        )
        
        # Initialize retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": DEFAULT_RETRIEVAL_K})
        
        # Build research workflow
        logger.info("🔧 Building research workflow...")
        research_agent = build_research_workflow()
        
        logger.info("✅ All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}")
        return False

# ===================================================================
# LANGRAPH STATE AND WORKFLOW
# ===================================================================

class GraphState(TypedDict):
    """State schema for the research workflow."""
    query: str
    log: List[str]
    documents: List[dict]
    rag_context: str
    answer: str
    error: Optional[str]

def planning_node(state: GraphState) -> Dict[str, Any]:
    """
    Generate search strategy for the research query.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with search query
    """
    try:
        query = state["query"]
        log = state.get("log", [])
        log.append("🧠 Planning research strategy...")
        
        if not query or len(query.strip()) == 0:
            error = "Empty research query provided"
            log.append(f"❌ {error}")
            return {"log": log, "error": error, "documents": []}
        
        if len(query) > MAX_QUERY_LENGTH:
            error = f"Query too long ({len(query)} chars, max {MAX_QUERY_LENGTH})"
            log.append(f"❌ {error}")
            return {"log": log, "error": error, "documents": []}
        
        # Generate search query using LLM
        prompt = ChatPromptTemplate.from_template(
            "Generate a concise, effective web search query for this research question.\n"
            "Focus on key terms and concepts that will find relevant information.\n"
            "Question: {question}\n"
            "Search query:"
        )
        
        try:
            chain = prompt | llm
            search_query = chain.invoke({"question": query}).content.strip()
            log.append(f"📝 Generated search query: '{search_query}'")
            
            return {
                "log": log, 
                "documents": [{"query": search_query}]
            }
            
        except Exception as e:
            error = f"Failed to generate search query: {str(e)}"
            log.append(f"❌ {error}")
            return {"log": log, "error": error, "documents": []}
        
    except Exception as e:
        logger.error(f"Planning node error: {e}")
        return {
            "log": ["❌ Planning failed"],
            "error": str(e),
            "documents": []
        }

def search_node(state: GraphState) -> Dict[str, Any]:
    """
    Perform web search using MCP client with comprehensive error handling.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with search results
    """
    try:
        if state.get("error"):
            return {"log": state.get("log", []), "documents": []}
            
        search_query = state["documents"][0]["query"]
        log = state.get("log", [])
        log.append(f"🔍 Searching the web: '{search_query}'")
        
        mcp_server_url = os.getenv("MCP_SERVER_URL")
        request_id = str(uuid.uuid4())
        
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
                response = client.post(
                    mcp_server_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "call_tool",
                        "params": {
                            "name": "web_search",
                            "args": {"query": search_query}
                        }
                    }
                )
                
                response.raise_for_status()
                
                # Parse and validate response
                try:
                    result_data = response.json()
                except json.JSONDecodeError as e:
                    error = f"Invalid JSON response from MCP server: {e}"
                    log.append(f"❌ {error}")
                    return {"log": log, "error": error, "documents": []}
                
                # Extract content safely
                content = extract_search_content(result_data, log)
                
                return {
                    "log": log,
                    "documents": [{"content": content}]
                }
                
        except httpx.ConnectError:
            error = "MCP server connection failed - check if server is running on port 8002"
            log.append(f"❌ {error}")
            return {"log": log, "error": error, "documents": []}
            
        except httpx.TimeoutException:
            error = f"MCP server request timed out after {DEFAULT_TIMEOUT}s"
            log.append(f"❌ {error}")
            return {"log": log, "error": error, "documents": []}
            
        except httpx.HTTPStatusError as e:
            error = f"MCP server HTTP error: {e.response.status_code}"
            log.append(f"❌ {error}")
            return {"log": log, "error": error, "documents": []}
            
        except Exception as e:
            error = f"Unexpected search error: {str(e)}"
            log.append(f"❌ {error}")
            logger.error(f"Search node error: {e}")
            return {"log": log, "error": error, "documents": []}
            
    except Exception as e:
        logger.error(f"Search node critical error: {e}")
        return {
            "log": ["❌ Search failed critically"],
            "error": str(e),
            "documents": []
        }

def extract_search_content(result_data: dict, log: list) -> str:
    """
    Safely extract content from MCP server response.
    
    Args:
        result_data: JSON response from MCP server
        log: Log list to append messages
        
    Returns:
        Extracted content string
    """
    try:
        if "error" in result_data:
            error_info = result_data["error"]
            error_msg = f"MCP server error: {error_info.get('message', 'Unknown error')}"
            log.append(f"❌ {error_msg}")
            return f"Search failed: {error_msg}"
        
        if "result" not in result_data:
            log.append("⚠️ No result field in MCP response")
            return "No search results available"
        
        result = result_data["result"]
        
        # Handle different response formats
        if isinstance(result, str):
            log.append("✅ Search completed")
            return result
        elif isinstance(result, dict):
            if "content" in result:
                content = result["content"]
                if isinstance(content, list) and content:
                    if isinstance(content[0], dict) and "text" in content[0]:
                        log.append("✅ Search completed")
                        return content[0]["text"]
                    else:
                        log.append("✅ Search completed (formatted)")
                        return str(content[0])
                else:
                    log.append("✅ Search completed (direct)")
                    return str(content)
            else:
                log.append("✅ Search completed (raw)")
                return str(result)
        else:
            log.append("✅ Search completed (string)")
            return str(result)
            
    except Exception as e:
        error_msg = f"Error extracting search content: {e}"
        log.append(f"❌ {error_msg}")
        return f"Content extraction failed: {error_msg}"

def process_and_chunk_node(state: GraphState) -> Dict[str, Any]:
    """
    Process and chunk search results for vector storage.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with chunked documents
    """
    try:
        if state.get("error"):
            return {"log": state.get("log", []), "documents": []}
            
        log = state.get("log", [])
        log.append("📄 Processing and chunking content...")
        
        raw_content = [doc["content"] for doc in state["documents"]]
        
        if not raw_content or not any(content.strip() for content in raw_content):
            log.append("⚠️ No content to process")
            return {"log": log, "documents": []}
        
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP
            )
            chunked_docs = text_splitter.create_documents(raw_content)
            
            log.append(f"✅ Created {len(chunked_docs)} document chunks")
            return {"log": log, "documents": chunked_docs}
            
        except Exception as e:
            error = f"Document chunking failed: {str(e)}"
            log.append(f"❌ {error}")
            return {"log": log, "error": error, "documents": []}
            
    except Exception as e:
        logger.error(f"Processing node error: {e}")
        return {
            "log": ["❌ Document processing failed"],
            "error": str(e),
            "documents": []
        }

def update_vector_store_node(state: GraphState) -> Dict[str, Any]:
    """
    Update vector store with new documents.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state
    """
    try:
        if state.get("error") or not state.get("documents"):
            return {"log": state.get("log", [])}
            
        log = state.get("log", [])
        log.append("💾 Updating knowledge base...")
        
        try:
            vectorstore.add_documents(state["documents"])
            log.append("✅ Knowledge base updated successfully")
            return {"log": log}
            
        except Exception as e:
            error = f"Vector store update failed: {str(e)}"
            log.append(f"❌ {error}")
            return {"log": log, "error": error}
            
    except Exception as e:
        logger.error(f"Vector store node error: {e}")
        return {
            "log": ["❌ Knowledge base update failed"],
            "error": str(e)
        }

def rag_retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve relevant context using RAG.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with retrieved context
    """
    try:
        if state.get("error"):
            return {"log": state.get("log", []), "rag_context": ""}
            
        log = state.get("log", [])
        log.append("🔍 Retrieving relevant context...")
        
        try:
            query = state["query"]
            retrieved_docs = retriever.invoke(query)
            
            if not retrieved_docs:
                log.append("⚠️ No relevant context found")
                return {"log": log, "rag_context": ""}
            
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            log.append(f"✅ Retrieved {len(retrieved_docs)} relevant documents")
            
            return {"log": log, "rag_context": context}
            
        except Exception as e:
            error = f"Context retrieval failed: {str(e)}"
            log.append(f"❌ {error}")
            return {"log": log, "error": error, "rag_context": ""}
            
    except Exception as e:
        logger.error(f"RAG retrieval node error: {e}")
        return {
            "log": ["❌ Context retrieval failed"],
            "error": str(e),
            "rag_context": ""
        }

def synthesis_node(state: GraphState) -> Dict[str, Any]:
    """
    Synthesize final answer using retrieved context.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with final answer
    """
    try:
        log = state.get("log", [])
        
        if state.get("error"):
            log.append("⚠️ Providing answer based on available information...")
            context = state.get("rag_context", "")
            if not context:
                answer = f"I apologize, but I encountered an error during research: {state['error']}. Please try rephrasing your question or check the system configuration."
                return {"log": log, "answer": answer}
        else:
            log.append("✍️ Synthesizing comprehensive answer...")
        
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are an expert research assistant. Use the provided context to answer the user's question comprehensively.\n"
                "If the context is insufficient, acknowledge this and provide what information you can.\n"
                "Include relevant details and cite sources when possible.\n\n"
                "CONTEXT:\n{context}\n\n"
                "QUESTION:\n{question}\n\n"
                "ANSWER:"
            )
            
            chain = prompt | llm
            context = state.get("rag_context", "")
            query = state["query"]
            
            final_answer = chain.invoke({
                "context": context if context else "No additional context available.",
                "question": query
            }).content
            
            log.append("🎉 Research completed successfully!")
            return {"log": log, "answer": final_answer}
            
        except Exception as e:
            error = f"Answer synthesis failed: {str(e)}"
            log.append(f"❌ {error}")
            fallback_answer = f"I apologize, but I encountered an error while generating the answer: {error}. Please try again or contact support."
            return {"log": log, "answer": fallback_answer, "error": error}
            
    except Exception as e:
        logger.error(f"Synthesis node error: {e}")
        return {
            "log": ["❌ Answer synthesis failed"],
            "answer": "I apologize, but I encountered a critical error. Please try again.",
            "error": str(e)
        }

def build_research_workflow() -> StateGraph:
    """
    Build the complete research workflow using LangGraph.
    
    Returns:
        Compiled workflow graph
    """
    try:
        workflow = StateGraph(GraphState)
        
        # Add all nodes
        workflow.add_node("planning", planning_node)
        workflow.add_node("search", search_node)
        workflow.add_node("process_and_chunk", process_and_chunk_node)
        workflow.add_node("update_vector_store", update_vector_store_node)
        workflow.add_node("rag_retrieval", rag_retrieval_node)
        workflow.add_node("synthesis", synthesis_node)
        
        # Define workflow edges
        workflow.set_entry_point("planning")
        workflow.add_edge("planning", "search")
        workflow.add_edge("search", "process_and_chunk")
        workflow.add_edge("process_and_chunk", "update_vector_store")
        workflow.add_edge("update_vector_store", "rag_retrieval")
        workflow.add_edge("rag_retrieval", "synthesis")
        workflow.add_edge("synthesis", END)
        
        return workflow.compile()
        
    except Exception as e:
        logger.error(f"Failed to build research workflow: {e}")
        raise

# ===================================================================
# FASTAPI APPLICATION
# ===================================================================

app = FastAPI(
    title="ME344 Deep Research Agent",
    description="LangGraph-powered multi-step research system with MCP integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ResearchRequest(BaseModel):
    """Request model for research queries."""
    query: str

    class Config:
        schema_extra = {
            "example": {
                "query": "What are the latest developments in artificial intelligence?"
            }
        }

# ===================================================================
# API ENDPOINTS
# ===================================================================

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "ME344 Deep Research Agent",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/research": "POST - Submit research query",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test core components
        if not all([llm, embedding_function, vectorstore, research_agent]):
            return {"status": "unhealthy", "error": "Components not initialized"}
        
        return {
            "status": "healthy",
            "components": {
                "llm": "✅ Ready",
                "embeddings": "✅ Ready", 
                "vectorstore": "✅ Ready",
                "workflow": "✅ Ready"
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def research_event_stream(query: str):
    """
    Stream research progress and results as Server-Sent Events.
    
    Args:
        query: Research query string
        
    Yields:
        SSE-formatted progress updates and final result
    """
    try:
        if not research_agent:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Research agent not initialized'})}\n\n"
            return
        
        # Validate query
        if not query or not query.strip():
            yield f"data: {json.dumps({'type': 'error', 'content': 'Empty query provided'})}\n\n"
            return
        
        if len(query) > MAX_QUERY_LENGTH:
            yield f"data: {json.dumps({'type': 'error', 'content': f'Query too long (max {MAX_QUERY_LENGTH} characters)'})}\n\n"
            return
        
        # Execute research workflow
        final_state = None
        async for chunk in research_agent.astream({"query": query.strip()}, stream_mode="values"):
            final_state = chunk
        
        if not final_state:
            yield f"data: {json.dumps({'type': 'error', 'content': 'No response from research workflow'})}\n\n"
            return
        
        # Stream log messages
        for log_message in final_state.get("log", []):
            yield f"data: {json.dumps({'type': 'log', 'content': log_message})}\n\n"
        
        # Stream final answer
        answer = final_state.get("answer", "No answer generated")
        if final_state.get("error"):
            yield f"data: {json.dumps({'type': 'error', 'content': final_state['error']})}\n\n"
        
        yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"
        
    except Exception as e:
        logger.error(f"Research stream error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Research failed: {str(e)}'})}\n\n"

@app.post("/research")
async def research_endpoint(request: ResearchRequest):
    """
    Main research endpoint that returns streaming response.
    
    Args:
        request: Research request with query
        
    Returns:
        StreamingResponse with Server-Sent Events
    """
    try:
        return StreamingResponse(
            research_event_stream(request.query),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        logger.error(f"Research endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================================================================
# MAIN EXECUTION
# ===================================================================

def main() -> None:
    """Main application entry point with comprehensive validation."""
    
    # Print startup information
    print_startup_info()
    
    # Validate environment
    logger.info("🔍 Validating environment...")
    is_valid, errors = validate_environment()
    
    if not is_valid:
        logger.error("❌ Environment validation failed:")
        for error in errors:
            logger.error(f"   {error}")
        
        print("\n❌ Deep Research Agent startup failed.")
        print("Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    logger.info("✅ Environment validation passed")
    
    # Start server
    try:
        import uvicorn
        logger.info(f"🚀 Starting Deep Research Agent on {SERVER_HOST}:{SERVER_PORT}")
        
        uvicorn.run(
            app,
            host=SERVER_HOST,
            port=SERVER_PORT,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()