#!/usr/bin/env python3
"""
ME344 Deep Research Agent
Performance-optimized multi-step research system with MCP integration

This agent orchestrates complex research workflows by combining web search,
document processing, vector storage, and RAG-based synthesis.

Key features:
- Fast response times (<1 minute target)
- Streaming responses for better UX
- Graceful timeout handling
- Performance monitoring
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        
        # Initialize vector store
        logger.info("🔧 Initializing vector store...")
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
    rag_context: str
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
        
        # OPTIMIZED: Shorter, more direct prompt
        prompt = ChatPromptTemplate.from_template(
            "Generate a web search query (max 50 chars) for: {question}\n"
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
            return {"log": state.get("log", []), "documents": []}
            
        search_query = state["documents"][0]["query"]
        request_id = state.get("request_id")
        log = state.get("log", [])
        log.append(f"🔍 Searching the web: '{search_query}'")
        
        # Use Tavily directly instead of MCP server
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            error = "TAVILY_API_KEY not set - web search unavailable"
            log.append(f"❌ {error}")
            return {"log": log, "error": error, "documents": []}
        
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
                return {"log": log, "error": error, "documents": []}
            
            return {
                "log": log,
                "documents": [{"content": content}]
            }
                
        except ImportError as e:
            error = f"Tavily library not installed: {e}"
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

def process_and_chunk_node(state: GraphState) -> Dict[str, Any]:
    """Process and chunk search results for vector storage."""
    logger.info("📄 PROCESS NODE: Starting")
    start_time = time.time()
    
    try:
        if state.get("error"):
            return {"log": state.get("log", []), "documents": []}
            
        request_id = state.get("request_id")
        log = state.get("log", [])
        log.append("📄 Processing and chunking content...")
        
        # Safely extract content from documents with error handling
        raw_content = []
        for doc in state.get("documents", []):
            if isinstance(doc, dict) and "content" in doc:
                raw_content.append(doc["content"])
            elif isinstance(doc, str):
                raw_content.append(doc)
            else:
                logger.warning(f"Skipping invalid document format: {type(doc)}")
        
        if not raw_content or not any(content.strip() for content in raw_content):
            log.append("⚠️ No content to process")
            return {"log": log, "documents": []}
        
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP
            )
            chunked_docs = text_splitter.create_documents(raw_content)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ PROCESS NODE: Complete in {elapsed:.2f}s")
            
            if request_id:
                request_logger.add_timing(request_id, "process", elapsed)
            
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
    """Update vector store with new documents."""
    logger.info("💾 VECTOR STORE NODE: Starting")
    start_time = time.time()
    
    try:
        if state.get("error") or not state.get("documents"):
            return {"log": state.get("log", [])}
            
        request_id = state.get("request_id")
        log = state.get("log", [])
        log.append("💾 Updating knowledge base...")
        
        try:
            vectorstore.add_documents(state["documents"])
            
            elapsed = time.time() - start_time
            logger.info(f"✅ VECTOR STORE NODE: Complete in {elapsed:.2f}s")
            
            if request_id:
                request_logger.add_timing(request_id, "vector_store", elapsed)
            
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
    """Retrieve relevant context using RAG."""
    logger.info("🔎 RAG RETRIEVAL NODE: Starting")
    start_time = time.time()
    
    try:
        if state.get("error"):
            return {"log": state.get("log", []), "rag_context": ""}
            
        request_id = state.get("request_id")
        log = state.get("log", [])
        log.append("🔍 Retrieving relevant context...")
        
        try:
            query = state["query"]
            retrieved_docs = retriever.invoke(query)
            
            if not retrieved_docs:
                log.append("⚠️ No relevant context found")
                return {"log": log, "rag_context": ""}
            
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            elapsed = time.time() - start_time
            logger.info(f"✅ RAG RETRIEVAL NODE: Complete in {elapsed:.2f}s")
            
            if request_id:
                request_logger.add_timing(request_id, "rag_retrieval", elapsed)
            
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

async def synthesis_node(state: GraphState) -> Dict[str, Any]:
    """Synthesize final answer using retrieved context - OPTIMIZED."""
    logger.info("✍️ SYNTHESIS NODE: Starting")
    start_time = time.time()
    
    try:
        request_id = state.get("request_id")
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
            # OPTIMIZED: Ultra-concise prompt for faster responses
            prompt = ChatPromptTemplate.from_template(
                "Answer in 2-3 sentences with key facts only. Be direct and specific.\n\n"
                "CONTEXT:\n{context}\n\n"
                "QUESTION: {question}\n\n"
                "ANSWER:"
            )
            
            chain = prompt | llm
            context = state.get("rag_context", "")
            query = state["query"]
            
            # Limit context size to avoid long processing
            if len(context) > 3000:
                context = context[:3000] + "..."
            
            logger.info(f"📤 SYNTHESIS NODE: Calling LLM (context: {len(context)} chars)")
            log.append(f"📝 Generating final answer (context: {len(context)} chars)...")
            
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
        # Create the same prompt as synthesis_node
        prompt = ChatPromptTemplate.from_template(
            "Answer in 2-3 sentences with key facts only. Be direct and specific.\n\n"
            "CONTEXT:\n{context}\n\n"
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
                node_order = ["planning", "search", "process_and_chunk", "update_vector_store", "rag_retrieval", "synthesis"]
                
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
        # Always attempt synthesis unless there's a critical error that prevents any answer
        if not final_state.get("error") or (final_state.get("documents") and len(final_state.get("documents", [])) > 0):
            # Use RAG context if available, otherwise use raw search results
            context = final_state.get("rag_context", "")
            
            # If no RAG context but we have search documents, create context from them
            if not context and final_state.get("documents"):
                logger.info(f"📝 No RAG context available, using search results for synthesis")
                search_docs = final_state.get("documents", [])
                context_parts = []
                for i, doc in enumerate(search_docs[:3], 1):  # Use top 3 search results
                    if isinstance(doc, dict) and "content" in doc:
                        context_parts.append(f"Result {i}:\n{doc['content'][:500]}...")
                    elif hasattr(doc, 'page_content'):
                        context_parts.append(f"Result {i}:\n{doc.page_content[:500]}...")
                context = "\n\n".join(context_parts)
                yield f"data: {json.dumps({'type': 'log', 'content': '⚠️ Using search results directly (RAG unavailable)'})}\n\n"
            
            # If still no context, provide a helpful message
            if not context:
                context = f"No specific context was found for the query: {query}. Please provide a general answer based on your knowledge."
                yield f"data: {json.dumps({'type': 'log', 'content': '⚠️ No context available, using general knowledge'})}\n\n"
            
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