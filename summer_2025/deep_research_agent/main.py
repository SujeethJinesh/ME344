#!/usr/bin/env python3
"""
ME344 Deep Research Agent
Streamlined web research system with direct search result synthesis

This agent performs web searches and synthesizes answers directly from
Tavily search results without intermediate storage.

Key features:
- Direct web search integration via Tavily API
- Streaming responses for better UX
- Source attribution with URLs
- Simplified 3-node workflow: planning → search → synthesis
"""

import os
import sys
import json
import logging
import httpx
import uuid
import asyncio
from typing import TypedDict, List, Dict, Any, Optional, AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from collections import deque
import threading
import time

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# from langchain_text_splitters import RecursiveCharacterTextSplitter  # Removed - no longer chunking
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
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

# OPTIMIZED TIMEOUTS
PLANNING_TIMEOUT = 30  # Reduced from unlimited
SYNTHESIS_TIMEOUT = 60  # Reduced from unlimited
WORKFLOW_TIMEOUT = 120  # Reduced from 300
DEFAULT_TIMEOUT = 30   # Reduced from 60

MAX_QUERY_LENGTH = 1000

# Environment configuration with validation
REQUIRED_ENV_VARS = {
    "MCP_SERVER_URL": "http://localhost:8002/mcp",
    "CHROMA_PERSIST_DIR": "./deep_research_chroma",
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "EMBEDDING_MODEL": "nomic-embed-text",
    "LLM_MODEL": "gemma3:1b"
}

# ===================================================================
# ENVIRONMENT VALIDATION (Same as original)
# ===================================================================

def validate_environment() -> tuple[bool, list[str]]:
    """Validate environment variables and system requirements."""
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
    print("⚡ Performance Settings:")
    print(f"   Planning Timeout: {PLANNING_TIMEOUT}s")
    print(f"   Synthesis Timeout: {SYNTHESIS_TIMEOUT}s")
    print(f"   Workflow Timeout: {WORKFLOW_TIMEOUT}s")
    print(f"   LLM Temperature: 0 (deterministic)")
    print()

# ===================================================================
# REQUEST LOGGING SYSTEM (Same as original)
# ===================================================================

class RequestLogger:
    """Thread-safe request logging system."""
    
    def __init__(self, max_requests: int = 100):
        self.max_requests = max_requests
        self.requests = deque(maxlen=max_requests)
        self.lock = threading.Lock()
    
    def log_request(self, request_id: str, query: str) -> Dict[str, Any]:
        """Start logging a new request."""
        with self.lock:
            request_log = {
                "id": request_id,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "logs": [],
                "status": "in_progress",
                "error": None,
                "answer": None,
                "timings": {}  # Added for performance tracking
            }
            self.requests.append(request_log)
            return request_log
    
    def add_log(self, request_id: str, message: str, level: str = "info"):
        """Add a log entry to a request."""
        with self.lock:
            for req in self.requests:
                if req["id"] == request_id:
                    req["logs"].append({
                        "timestamp": datetime.now().isoformat(),
                        "level": level,
                        "message": message
                    })
                    break
    
    def add_timing(self, request_id: str, node: str, duration: float):
        """Add timing information for a node."""
        with self.lock:
            for req in self.requests:
                if req["id"] == request_id:
                    req["timings"][node] = duration
                    break
    
    def complete_request(self, request_id: str, answer: str = None, error: str = None):
        """Mark a request as complete."""
        with self.lock:
            for req in self.requests:
                if req["id"] == request_id:
                    req["status"] = "error" if error else "completed"
                    req["error"] = error
                    req["answer"] = answer
                    req["completed_at"] = datetime.now().isoformat()
                    break
    
    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific request log."""
        with self.lock:
            for req in self.requests:
                if req["id"] == request_id:
                    return req.copy()
            return None
    
    def get_all_requests(self) -> List[Dict[str, Any]]:
        """Get all request logs."""
        with self.lock:
            return list(self.requests)

# Global request logger
request_logger = RequestLogger()

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
    """Initialize all required components with error handling."""
    global llm, embedding_function, vectorstore, retriever, research_agent
    
    try:
        # Initialize LLM with optimized settings
        logger.info("🔧 Initializing LLM with optimized settings...")
        llm_model = os.getenv('LLM_MODEL', 'llama3.1')
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        logger.info(f"Using LLM model: {llm_model} at {ollama_url}")
        
        # OPTIMIZATION: Temperature 0 for faster, deterministic responses
        llm = ChatOllama(
            model=llm_model,
            base_url=ollama_url,
            temperature=0,
            num_predict=500,  # Limit response length
            timeout=60  # Add timeout
        )
        
        # Initialize embeddings
        logger.info("🔧 Initializing embeddings...")
        embedding_model = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
        logger.info(f"Using embedding model: {embedding_model} at {ollama_url}")
        embedding_function = OllamaEmbeddings(
            model=embedding_model,
            base_url=ollama_url
        )
        
        # Initialize vector store (kept for Part 1 compatibility)
        logger.info("🔧 Initializing vector store (for Part 1 compatibility)...")
        chroma_dir = os.getenv('CHROMA_PERSIST_DIR', './deep_research_chroma')
        logger.info(f"Using ChromaDB directory: {chroma_dir}")
        
        # Ensure the directory exists
        os.makedirs(chroma_dir, exist_ok=True)
        
        # Use environment variable for collection name or default to a consistent name
        collection_name = os.getenv('RESEARCH_COLLECTION_NAME', 'deep_research_collection')
        logger.info(f"Using collection name: {collection_name}")
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=chroma_dir,
        )
        
        # Initialize retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": DEFAULT_RETRIEVAL_K})
        
        # Build research workflow
        logger.info("🔧 Building optimized research workflow...")
        research_agent = build_research_workflow()
        
        # Test the workflow compilation
        logger.info("🧪 Testing workflow compilation...")
        if research_agent is None:
            logger.error("❌ Research workflow is None")
            return False
            
        logger.info("✅ All components initialized successfully")
        logger.info(f"   - LLM: {llm_model} (temp=0, optimized)")
        logger.info(f"   - Embeddings: {embedding_model}")
        logger.info(f"   - Collection: {collection_name}")
        logger.info(f"   - ChromaDB: {chroma_dir}")
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
    search_results: List[dict]  # Raw Tavily search results
    rag_context: str  # Keeping for backward compatibility
    answer: str
    error: Optional[str]
    request_id: Optional[str]  # Added for tracking

async def planning_node(state: GraphState) -> Dict[str, Any]:
    """Generate search strategy for the research query - OPTIMIZED."""
    logger.info("🎯 PLANNING NODE: Starting")
    start_time = time.time()
    
    try:
        query = state["query"]
        request_id = state.get("request_id")
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
        
        # Check if LLM is initialized
        if llm is None:
            error = "LLM not initialized"
            log.append(f"❌ {error}")
            logger.error("❌ PLANNING NODE: LLM is None!")
            return {"log": log, "error": error, "documents": []}
        
        # OPTIMIZED: Better search query generation
        prompt = ChatPromptTemplate.from_template(
            "Generate a simple web search query for: {question}\n"
            "Rules: No quotes, max 50 chars, use common keywords\n"
            "Query:"
        )
        
        try:
            logger.info(f"📤 PLANNING NODE: Calling LLM with timeout {PLANNING_TIMEOUT}s")
            chain = prompt | llm
            
            # Use asyncio timeout for better control
            async def call_llm():
                return await chain.ainvoke({"question": query})
            
            # Run with timeout
            result = await asyncio.wait_for(call_llm(), timeout=PLANNING_TIMEOUT)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ PLANNING NODE: LLM call completed in {elapsed:.2f}s")
            
            if request_id:
                request_logger.add_timing(request_id, "planning", elapsed)
            
            if result is None or not hasattr(result, 'content'):
                error = "LLM returned invalid response"
                log.append(f"❌ {error}")
                logger.error(f"❌ PLANNING NODE: {error} - result: {result}")
                return {"log": log, "error": error, "documents": []}
                
            search_query = result.content.strip()
            
            # Ensure query is concise
            if len(search_query) > 100:
                search_query = search_query[:97] + "..."
            
            log.append(f"📝 Generated search query: '{search_query}' ({len(search_query)} chars)")
            
            logger.info(f"✅ PLANNING NODE: Complete - Query: '{search_query}'")
            return {
                "log": log, 
                "documents": [{"query": search_query}]
            }
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            error = f"Planning timeout after {elapsed:.1f}s (limit: {PLANNING_TIMEOUT}s)"
            log.append(f"❌ {error}")
            logger.error(f"❌ PLANNING NODE: {error}")
            
            # Fallback: Use original query truncated
            fallback_query = query[:50] + "..." if len(query) > 50 else query
            log.append(f"⚠️ Using fallback query: '{fallback_query}'")
            return {
                "log": log,
                "documents": [{"query": fallback_query}]
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
    """Perform web search using MCP client - Same as original but with timing."""
    logger.info("🔍 SEARCH NODE: Starting")
    start_time = time.time()
    
    try:
        if state.get("error"):
            return {"log": state.get("log", []), "documents": [], "search_results": []}
            
        search_query = state["documents"][0]["query"]
        request_id = state.get("request_id")
        log = state.get("log", [])
        log.append(f"🔍 Searching the web: '{search_query}'")
        
        # Use Tavily directly instead of MCP server
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            error = "TAVILY_API_KEY not set - web search unavailable"
            log.append(f"❌ {error}")
            return {"log": log, "error": error, "documents": [], "search_results": []}
        
        try:
            # Import Tavily client
            from tavily import TavilyClient
            
            # Initialize Tavily client
            tavily = TavilyClient(api_key=tavily_api_key)
            
            # Perform search
            logger.info(f"📤 SEARCH NODE: Calling Tavily API with query: '{search_query}'")
            search_results = tavily.search(query=search_query, max_results=5)
            
            # Log the raw search results for debugging
            logger.info(f"📥 SEARCH NODE: Tavily returned {len(search_results.get('results', []))} results")
            for i, result in enumerate(search_results.get('results', []), 1):
                logger.info(f"  Result {i}: {result.get('title', 'No title')[:80]}...")
                logger.info(f"    URL: {result.get('url', 'No URL')}")
                logger.info(f"    Content preview: {result.get('content', 'No content')[:100]}...")
            
            # Format results
            content = f"🔍 Search Results for: '{search_query}'\n"
            content += f"📊 Found {len(search_results.get('results', []))} results\n"
            content += "=" * 50 + "\n\n"
            
            for i, result in enumerate(search_results.get('results', []), 1):
                content += f"[{i}] {result.get('title', 'No title')}\n"
                content += f"🔗 {result.get('url', 'No URL')}\n"
                content += f"📝 {result.get('content', 'No content available')}\n"
                content += "-" * 40 + "\n\n"
            
            elapsed = time.time() - start_time
            logger.info(f"✅ SEARCH NODE: Complete in {elapsed:.2f}s")
            
            if request_id:
                request_logger.add_timing(request_id, "search", elapsed)
            
            # Check if the content indicates an error
            if content.startswith("Error:") or content.startswith("Search failed:"):
                error = f"Search failed: {content}"
                log.append(f"❌ {error}")
                return {"log": log, "error": error, "documents": [], "search_results": []}
            
            # Return both raw results and formatted content
            return {
                "log": log,
                "documents": [{"content": content}],
                "search_results": search_results.get('results', [])
            }
                
        except ImportError as e:
            error = f"Tavily library not installed: {e}"
            log.append(f"❌ {error}")
            return {"log": log, "error": error, "documents": [], "search_results": []}
            
        except Exception as e:
            error = f"Unexpected search error: {str(e)}"
            log.append(f"❌ {error}")
            logger.error(f"Search node error: {e}")
            return {"log": log, "error": error, "documents": [], "search_results": []}
            
    except Exception as e:
        logger.error(f"Search node critical error: {e}")
        return {
            "log": ["❌ Search failed critically"],
            "error": str(e),
            "documents": [],
            "search_results": []
        }

def extract_search_content(result_data: dict, log: list) -> str:
    """Safely extract content from MCP server response."""
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

# NOTE: Removed process_and_chunk_node, update_vector_store_node, and rag_retrieval_node
# These are no longer needed as we use search results directly without RAG storage

async def synthesis_node(state: GraphState) -> Dict[str, Any]:
    """Synthesize final answer using search results directly."""
    logger.info("✍️ SYNTHESIS NODE: Starting")
    start_time = time.time()
    
    try:
        request_id = state.get("request_id")
        log = state.get("log", [])
        search_results = state.get("search_results", [])
        
        # Build context from search results
        if not search_results:
            if state.get("error"):
                log.append("⚠️ No search results available due to error")
                answer = f"I apologize, but I couldn't find any information due to: {state['error']}. Please try rephrasing your question."
            else:
                log.append("⚠️ No search results found")
                answer = "I couldn't find any relevant information for your query. Please try with different keywords or a more specific question."
            return {"log": log, "answer": answer}
        
        log.append("✍️ Synthesizing answer from web search results...")
        
        try:
            # Build context from search results with source attribution
            context_parts = []
            for i, result in enumerate(search_results[:5], 1):  # Use top 5 results
                title = result.get('title', 'No title')
                content = result.get('content', '')
                url = result.get('url', '')
                
                if content:
                    context_parts.append(f"[Source {i}] {title}\n{content}\nURL: {url}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Enhanced prompt for better synthesis with sources
            prompt = ChatPromptTemplate.from_template(
                "Based on the web search results below, provide a comprehensive answer in 2-3 sentences. "
                "Include the most relevant facts and cite source numbers [1], [2], etc.\n\n"
                "SEARCH RESULTS:\n{context}\n\n"
                "QUESTION: {question}\n\n"
                "ANSWER:"
            )
            
            chain = prompt | llm
            query = state["query"]
            
            # Limit context size to avoid long processing
            if len(context) > 5000:
                context = context[:5000] + "..."
            
            logger.info(f"📤 SYNTHESIS NODE: Calling LLM (context: {len(context)} chars, {len(search_results)} results)")
            log.append(f"📝 Generating final answer from {len(search_results)} search results...")
            
            try:
                # Direct LLM call without timeout for streaming compatibility
                result = await chain.ainvoke({
                    "context": context if context else "No additional context available.",
                    "question": query
                })
                final_answer = result.content
                
                elapsed = time.time() - start_time
                logger.info(f"✅ SYNTHESIS NODE: LLM call completed in {elapsed:.2f}s")
                
                if request_id:
                    request_logger.add_timing(request_id, "synthesis", elapsed)
                
            except Exception as e:
                # Handle any LLM errors
                elapsed = time.time() - start_time
                logger.error(f"❌ SYNTHESIS NODE: LLM error after {elapsed:.2f}s: {e}")
                
                # Provide a fallback answer based on context
                if context:
                    final_answer = (
                        f"Based on my research, here's what I found:\n\n"
                        f"{context[:500]}...\n\n"
                        f"(Error during synthesis: {str(e)})"
                    )
                else:
                    final_answer = (
                        f"I apologize, but I encountered an error: {str(e)}. "
                        "Please try with a different query."
                    )
                
                log.append(f"⚠️ Synthesis error after {elapsed:.1f}s")
            
            log.append("🎉 Research completed successfully!")
            logger.info("✅ SYNTHESIS NODE: Complete")
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

async def streaming_synthesis(context: str, question: str, request_id: Optional[str] = None):
    """Stream synthesis response token by token."""
    try:
        # Enhanced prompt for synthesis with source citations
        prompt = ChatPromptTemplate.from_template(
            "Based on the provided information, answer in 2-3 sentences with key facts. "
            "Be direct and specific. If sources are numbered, cite them.\n\n"
            "INFORMATION:\n{context}\n\n"
            "QUESTION: {question}\n\n"
            "ANSWER:"
        )
        
        chain = prompt | llm
        
        # Limit context size
        if len(context) > 3000:
            context = context[:3000] + "..."
        
        logger.info(f"🔄 Starting streaming synthesis for request {request_id}")
        
        # Stream the response
        token_count = 0
        async for chunk in chain.astream({
            "context": context if context else "No additional context available.",
            "question": question
        }):
            if hasattr(chunk, 'content') and chunk.content:
                token_count += 1
                yield {
                    "type": "partial_answer",
                    "content": chunk.content,
                    "token_count": token_count
                }
        
        logger.info(f"✅ Streaming synthesis completed with {token_count} tokens")
        yield {
            "type": "synthesis_complete",
            "token_count": token_count
        }
        
    except Exception as e:
        logger.error(f"Streaming synthesis error: {e}")
        yield {
            "type": "synthesis_error",
            "error": str(e)
        }

def build_research_workflow() -> StateGraph:
    """Build the complete research workflow using LangGraph."""
    try:
        workflow = StateGraph(GraphState)
        
        # Add only essential nodes for web research
        workflow.add_node("planning", planning_node)
        workflow.add_node("search", search_node)
        workflow.add_node("synthesis", synthesis_node)
        
        # Simplified workflow: planning -> search -> synthesis
        workflow.set_entry_point("planning")
        workflow.add_edge("planning", "search")
        workflow.add_edge("search", "synthesis")
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the latest developments in artificial intelligence?"
            }
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
            "/health": "GET - Health check",
            "/logs": "GET - View request logs",
            "/logs/{request_id}": "GET - View specific request log"
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
            },
            "performance": {
                "planning_timeout": f"{PLANNING_TIMEOUT}s",
                "synthesis_timeout": f"{SYNTHESIS_TIMEOUT}s",
                "workflow_timeout": f"{WORKFLOW_TIMEOUT}s"
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def research_event_stream(query: str):
    """Stream research progress and results as Server-Sent Events - OPTIMIZED."""
    request_id = str(uuid.uuid4())
    
    try:
        # Log the request
        request_logger.log_request(request_id, query)
        logger.info(f"🆔 Starting research request {request_id}: '{query}'")
        
        # Send request ID to client
        yield f"data: {json.dumps({'type': 'request_id', 'content': request_id})}\n\n"
        
        if not research_agent:
            error_msg = 'Research agent not initialized'
            request_logger.add_log(request_id, error_msg, "error")
            request_logger.complete_request(request_id, error=error_msg)
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            return
        
        # Validate query
        if not query or not query.strip():
            error_msg = 'Empty query provided'
            request_logger.add_log(request_id, error_msg, "error")
            request_logger.complete_request(request_id, error=error_msg)
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            return
        
        if len(query) > MAX_QUERY_LENGTH:
            error_msg = f'Query too long (max {MAX_QUERY_LENGTH} characters)'
            request_logger.add_log(request_id, error_msg, "error")
            request_logger.complete_request(request_id, error=error_msg)
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            return
        
        request_logger.add_log(request_id, "Starting research workflow", "info")
        
        # Execute research workflow with reduced timeout
        final_state = None
        
        try:
            request_logger.add_log(request_id, f"Executing workflow with {WORKFLOW_TIMEOUT}s timeout", "info")
            logger.info(f"🔄 Starting LangGraph workflow for request {request_id}")
            
            # Create async timeout context
            async with asyncio.timeout(WORKFLOW_TIMEOUT):
                step_count = 0
                node_order = ["planning", "search", "synthesis"]
                
                # Pass request_id in initial state
                initial_state = {
                    "query": query.strip(),
                    "request_id": request_id
                }
                
                async for chunk in research_agent.astream(initial_state, stream_mode="values"):
                    step_count += 1
                    final_state = chunk
                    
                    # Determine current node
                    current_node = "unknown"
                    if step_count <= len(node_order):
                        current_node = node_order[step_count - 1]
                    
                    # Log detailed chunk info for debugging
                    logger.info(f"📍 Request {request_id} - Step {step_count}: Node '{current_node}'")
                    
                    # Log specific node progress
                    if "log" in chunk and chunk["log"]:
                        latest_log = chunk["log"][-1] if isinstance(chunk["log"], list) else chunk["log"]
                        request_logger.add_log(request_id, f"[{current_node}] {latest_log}", "info")
                    else:
                        request_logger.add_log(request_id, f"Workflow step {step_count}: {current_node}", "info")
                    
                    # Stream progress update
                    yield f"data: {json.dumps({'type': 'progress', 'content': f'Processing {current_node}...'})}\n\n"
                    
        except asyncio.TimeoutError:
            error_msg = f'Workflow timeout after {WORKFLOW_TIMEOUT} seconds'
            logger.error(f"❌ Request {request_id} timed out")
            request_logger.add_log(request_id, error_msg, "error")
            
            # Provide partial answer if available
            if final_state and final_state.get("rag_context"):
                partial_answer = (
                    "I found relevant information but couldn't complete the full analysis in time. "
                    f"Here's what I found:\n\n{final_state['rag_context'][:500]}..."
                )
                yield f"data: {json.dumps({'type': 'answer', 'content': partial_answer})}\n\n"
                request_logger.complete_request(request_id, answer=partial_answer, error=error_msg)
            else:
                request_logger.complete_request(request_id, error=error_msg)
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            return
            
        except Exception as e:
            error_msg = f'Workflow execution error: {str(e)}'
            logger.error(f"❌ Request {request_id} workflow error: {e}")
            request_logger.add_log(request_id, error_msg, "error")
            request_logger.complete_request(request_id, error=error_msg)
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            return
        
        if not final_state:
            error_msg = 'No response from research workflow'
            request_logger.add_log(request_id, error_msg, "error")
            request_logger.complete_request(request_id, error=error_msg)
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            return
        
        # Stream log messages
        for log_message in final_state.get("log", []):
            request_logger.add_log(request_id, log_message, "info")
            yield f"data: {json.dumps({'type': 'log', 'content': log_message})}\n\n"
        
        # Initialize answer variable for later use
        answer = ""
        
        # Check if we should stream the synthesis
        # Use search results directly for synthesis
        search_results = final_state.get("search_results", [])
        
        if search_results or not final_state.get("error"):
            # Build context from search results
            context = ""
            if search_results:
                logger.info(f"📝 Building context from {len(search_results)} search results")
                context_parts = []
                for i, result in enumerate(search_results[:5], 1):  # Use top 5 results
                    title = result.get('title', 'No title')
                    content = result.get('content', '')
                    url = result.get('url', '')
                    
                    if content:
                        context_parts.append(f"[Source {i}] {title}\n{content}\nURL: {url}")
                
                context = "\n\n---\n\n".join(context_parts)
                yield f"data: {json.dumps({'type': 'log', 'content': f'📚 Using {len(search_results)} web search results'})}\n\n"
            
            # If no search results, provide a helpful message
            if not context:
                context = f"No search results were found for: {query}. Please try with different keywords or a more specific question."
                yield f"data: {json.dumps({'type': 'log', 'content': '⚠️ No search results available'})}\n\n"
            
            query = query.strip()
            
            logger.info(f"📝 Starting streaming synthesis for request {request_id}")
            yield f"data: {json.dumps({'type': 'log', 'content': '🔄 Generating answer...'})}\n\n"
            
            full_answer = ""
            synthesis_error = False
            async for chunk in streaming_synthesis(context, query, request_id):
                if chunk["type"] == "partial_answer":
                    full_answer += chunk["content"]
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif chunk["type"] == "synthesis_complete":
                    token_msg = f"✅ Generated {chunk['token_count']} tokens"
                    yield f"data: {json.dumps({'type': 'log', 'content': token_msg})}\n\n"
                elif chunk["type"] == "synthesis_error":
                    synthesis_error = True
                    error_msg = f"Synthesis error: {chunk['error']}"
                    yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                    # If synthesis failed, use any existing answer or provide fallback
                    if not full_answer:
                        full_answer = final_state.get("answer", "I encountered an error during synthesis. Please try again.")
            
            # Only send answer if we have content
            if full_answer or not synthesis_error:
                # Send complete answer event to mark end of streaming
                yield f"data: {json.dumps({'type': 'answer', 'content': full_answer})}\n\n"
                final_state["answer"] = full_answer
            
            # Set answer for logging
            answer = full_answer
        else:
            # Critical error - provide best available answer
            logger.warning(f"Cannot perform synthesis for request {request_id}, using fallback")
            
            # Try to construct a meaningful answer from available data
            answer = final_state.get("answer", "")
            
            if not answer:
                if final_state.get("error"):
                    # Check if we have any partial results
                    if final_state.get("documents"):
                        answer = (
                            f"I encountered an error during processing: {final_state['error']}\n\n"
                            f"However, I was able to find some information:\n\n"
                            f"{str(final_state['documents'][0])[:300]}...\n\n"
                            f"Please try again or rephrase your query for better results."
                        )
                    else:
                        answer = (
                            f"I apologize, but I encountered an error: {final_state['error']}\n\n"
                            f"Please try again with a different query or check the system status."
                        )
                else:
                    answer = "I couldn't generate a complete answer. Please try rephrasing your query."
            
            if final_state.get("error"):
                yield f"data: {json.dumps({'type': 'error', 'content': final_state['error']})}\n\n"
            
            yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"
        
        # Add performance summary
        timings = request_logger.get_request(request_id).get("timings", {})
        if timings:
            total_time = sum(timings.values())
            perf_summary = f"⏱️ Performance: Total {total_time:.1f}s"
            for node, duration in timings.items():
                perf_summary += f", {node}: {duration:.1f}s"
            yield f"data: {json.dumps({'type': 'log', 'content': perf_summary})}\n\n"
        
        # Complete the request log
        if final_state.get("error"):
            request_logger.complete_request(request_id, answer=answer, error=final_state['error'])
        else:
            request_logger.complete_request(request_id, answer=answer)
        
        logger.info(f"✅ Completed research request {request_id}")
        
    except Exception as e:
        logger.error(f"Research stream error: {e}")
        request_logger.add_log(request_id, f"Critical error: {str(e)}", "error")
        request_logger.complete_request(request_id, error=str(e))
        yield f"data: {json.dumps({'type': 'error', 'content': f'Research failed: {str(e)}'})}\n\n"

@app.post("/research")
async def research_endpoint(request: ResearchRequest):
    """Main research endpoint that returns streaming response."""
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

@app.get("/logs")
async def get_all_logs():
    """Get all request logs with performance metrics."""
    return JSONResponse(content={
        "logs": request_logger.get_all_requests(),
        "total": len(request_logger.requests)
    })

@app.get("/logs/{request_id}")
async def get_request_log(request_id: str):
    """Get logs for a specific request with performance breakdown."""
    log = request_logger.get_request(request_id)
    if not log:
        raise HTTPException(status_code=404, detail="Request not found")
    return JSONResponse(content=log)

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