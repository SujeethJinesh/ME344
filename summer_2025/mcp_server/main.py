#!/usr/bin/env python3
"""
ME344 MCP (Model Context Protocol) Server
Provides web search capabilities via Tavily API for the Deep Research Agent

This server implements the MCP protocol to expose web search tools
that can be called by other services in the multi-agent system.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from langchain_community.tools.tavily_search import TavilySearchResults

# ===================================================================
# CONFIGURATION AND LOGGING SETUP
# ===================================================================

# Configure logging with consistent format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_NAME = "ME344 Web Research Tools Server"
SERVER_DESCRIPTION = "A server providing web search capabilities via Tavily API for RAG systems"
DEFAULT_PORT = 8002
DEFAULT_SEARCH_RESULTS = 3

# ===================================================================
# ENVIRONMENT VALIDATION
# ===================================================================

def validate_environment() -> tuple[bool, list[str]]:
    """
    Validate required environment variables and system requirements.
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check for required API key
    if not os.getenv("TAVILY_API_KEY"):
        errors.append("TAVILY_API_KEY environment variable not set")
    
    # Check Python version
    if sys.version_info < (3, 8):
        errors.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check if we can import required packages
    try:
        import mcp.server.fastmcp
        import langchain_community.tools.tavily_search
    except ImportError as e:
        errors.append(f"Missing required package: {e}")
        errors.append("  Solution: pip install -r requirements.txt")
    
    return len(errors) == 0, errors

def print_startup_info() -> None:
    """Print startup information and configuration."""
    print("=" * 70)
    print("🔍 ME344 MCP Web Search Server")
    print("   Tavily API Integration for Deep Research Agent")
    print("=" * 70)
    print()
    
    # Environment info
    print("🔧 Configuration:")
    print(f"   API Key: {'✅ Set' if os.getenv('TAVILY_API_KEY') else '❌ Missing'}")
    print(f"   Port: {DEFAULT_PORT}")
    print(f"   Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print()

# ===================================================================
# MCP SERVER SETUP
# ===================================================================

def create_mcp_server() -> Optional[FastMCP]:
    """
    Create and configure the MCP server with error handling.
    
    Returns:
        FastMCP instance or None if creation fails
    """
    try:
        mcp = FastMCP(
            SERVER_NAME,
            description=SERVER_DESCRIPTION
        )
        logger.info("MCP server instance created successfully")
        return mcp
    except Exception as e:
        logger.error(f"Failed to create MCP server: {e}")
        return None

def create_tavily_tool() -> Optional[TavilySearchResults]:
    """
    Create Tavily search tool with error handling.
    
    Returns:
        TavilySearchResults instance or None if creation fails
    """
    try:
        tavily_tool = TavilySearchResults(k=DEFAULT_SEARCH_RESULTS)
        logger.info("Tavily search tool initialized successfully")
        return tavily_tool
    except Exception as e:
        logger.error(f"Failed to initialize Tavily tool: {e}")
        logger.error("Check your TAVILY_API_KEY and network connection")
        return None

# ===================================================================
# WEB SEARCH TOOL IMPLEMENTATION
# ===================================================================

def setup_web_search_tool(mcp: FastMCP, tavily_tool: TavilySearchResults) -> None:
    """Setup the web search tool with comprehensive error handling."""
    
    @mcp.tool()
    def web_search(query: str) -> str:
        """
        Performs a web search using the Tavily API to find relevant information.
        
        Args:
            query: The search query string (max 500 characters)
            
        Returns:
            A formatted string containing the search results with metadata
            
        Raises:
            ValueError: If query is empty or too long
            RuntimeError: If search service is unavailable
        """
        # Input validation
        if not query or not query.strip():
            error_msg = "Search query cannot be empty"
            logger.warning(error_msg)
            return f"Error: {error_msg}"
        
        query = query.strip()
        if len(query) > 500:
            error_msg = "Search query too long (max 500 characters)"
            logger.warning(f"{error_msg}: {len(query)} characters")
            return f"Error: {error_msg}"
        
        logger.info(f"🔍 Processing search request: '{query}'")
        
        try:
            # Perform the search
            results = tavily_tool.invoke(query)
            
            if not results:
                logger.warning(f"No results found for query: '{query}'")
                return "No search results found for the given query."
            
            logger.info(f"✅ Found {len(results)} search results")
            
            # Format results for better readability
            formatted_results = format_search_results(results, query)
            return formatted_results
            
        except Exception as e:
            error_msg = f"Search service error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # Provide helpful error messages based on error type
            if "api" in str(e).lower() or "key" in str(e).lower():
                return f"Error: API authentication failed. Please check your TAVILY_API_KEY."
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                return f"Error: Network connection failed. Please check your internet connection."
            elif "timeout" in str(e).lower():
                return f"Error: Search request timed out. Please try a simpler query."
            else:
                return f"Error: {error_msg}"

def format_search_results(results: list, query: str) -> str:
    """
    Format search results into a readable string with metadata.
    
    Args:
        results: List of search results from Tavily
        query: Original search query
        
    Returns:
        Formatted string with search results and metadata
    """
    try:
        formatted = f"🔍 Search Results for: '{query}'\n"
        formatted += f"📊 Found {len(results)} results\n"
        formatted += "=" * 50 + "\n\n"
        
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                title = result.get('title', 'No title')
                url = result.get('url', 'No URL')
                content = result.get('content', 'No content available')
                
                formatted += f"{i}. {title}\n"
                formatted += f"   🔗 {url}\n"
                formatted += f"   📝 {content[:200]}{'...' if len(content) > 200 else ''}\n\n"
            else:
                # Handle string results
                formatted += f"{i}. {str(result)[:300]}{'...' if len(str(result)) > 300 else ''}\n\n"
        
        formatted += "=" * 50 + "\n"
        formatted += f"✅ Search completed successfully"
        
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting search results: {e}")
        return f"Search results: {str(results)}"

# ===================================================================
# MAIN SERVER EXECUTION
# ===================================================================

def main() -> None:
    """Main server execution with comprehensive error handling."""
    
    # Print startup information
    print_startup_info()
    
    # Validate environment
    logger.info("🔍 Validating environment...")
    is_valid, errors = validate_environment()
    
    if not is_valid:
        logger.error("❌ Environment validation failed:")
        for error in errors:
            if error.startswith("  "):
                logger.info(error)  # Solutions use info level
            else:
                logger.error(f"   {error}")
        
        print("\n❌ Server startup failed due to environment issues.")
        print("Please fix the above issues and try again.")
        sys.exit(1)
    
    logger.info("✅ Environment validation passed")
    
    # Create MCP server
    logger.info("🚀 Creating MCP server...")
    mcp = create_mcp_server()
    if not mcp:
        logger.error("❌ Failed to create MCP server")
        sys.exit(1)
    
    # Initialize Tavily tool
    logger.info("🔧 Initializing Tavily search tool...")
    tavily_tool = create_tavily_tool()
    if not tavily_tool:
        logger.error("❌ Failed to initialize Tavily tool")
        logger.error("Check your TAVILY_API_KEY and try again")
        sys.exit(1)
    
    # Setup web search tool
    logger.info("⚙️  Setting up web search tool...")
    try:
        setup_web_search_tool(mcp, tavily_tool)
        logger.info("✅ Web search tool configured")
    except Exception as e:
        logger.error(f"❌ Failed to setup web search tool: {e}")
        sys.exit(1)
    
    # Start the server
    try:
        logger.info(f"🌐 Starting MCP server...")
        print(f"🎉 MCP Server ready")
        print("📡 Available tools: web_search")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 70)
        
        # Use stdio transport (default for MCP)
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested by user")
        print("\n✅ MCP Server stopped gracefully")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        print(f"\n❌ Server failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()