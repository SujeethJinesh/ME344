#!/usr/bin/env python3
"""
ME344 Environment Validation Script
Comprehensive validation of system requirements and configuration

This script checks all prerequisites for running the Deep Research Agent system
and provides detailed guidance for fixing any issues found.
"""

import os
import sys
import subprocess
import json
import importlib.util
import socket
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ===================================================================
# CONFIGURATION
# ===================================================================

SCRIPT_VERSION = "1.0.0"
REQUIRED_PYTHON_VERSION = (3, 8)
REQUIRED_PORTS = {
    3000: "React Frontend",
    8000: "ChromaDB",
    8001: "Deep Research Agent", 
    8002: "MCP Server",
    11434: "Ollama"
}

REQUIRED_COMMANDS = {
    "python": "Python interpreter",
    "pip": "Python package manager",
    "node": "Node.js runtime",
    "npm": "Node.js package manager",
    "ollama": "Ollama AI runtime",
    "chroma": "ChromaDB server"
}

REQUIRED_PYTHON_PACKAGES = [
    ("fastapi", "FastAPI web framework"),
    ("uvicorn", "ASGI server"),
    ("langchain", "LangChain framework"),
    ("langchain_community", "LangChain community tools"),
    ("chromadb", "ChromaDB vector database"),
    ("httpx", "HTTP client"),
    ("langgraph", "LangGraph workflow orchestration"),
    ("tavily-python", "Tavily search API")
]

REQUIRED_DIRECTORIES = [
    "mcp_server",
    "deep_research_agent", 
    "llm-rag-chat",
    "data"
]

REQUIRED_FILES = [
    "run_part2.sh",
    "requirements.txt",
    "mcp_server/main.py",
    "deep_research_agent/main.py",
    "llm-rag-chat/package.json"
]

# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"{Colors.CYAN}{title}{Colors.END}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.END}")

def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def print_suggestion(message: str) -> None:
    """Print a suggestion message."""
    print(f"{Colors.PURPLE}💡 {message}{Colors.END}")

def run_command(command: str, capture_output: bool = True) -> Tuple[bool, str]:
    """
    Run a shell command and return success status and output.
    
    Args:
        command: Command to execute
        capture_output: Whether to capture command output
        
    Returns:
        Tuple of (success, output)
    """
    try:
        result = subprocess.run(
            command.split(),
            capture_output=capture_output,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False, ""

def check_port_available(port: int) -> bool:
    """
    Check if a port is available.
    
    Args:
        port: Port number to check
        
    Returns:
        True if port is available, False if in use
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            return result != 0  # Port is available if connection fails
    except Exception:
        return True

def check_internet_connection() -> bool:
    """Check if internet connection is available."""
    try:
        urllib.request.urlopen('https://www.google.com', timeout=5)
        return True
    except Exception:
        return False

# ===================================================================
# VALIDATION FUNCTIONS
# ===================================================================

def validate_python_version() -> Tuple[bool, List[str]]:
    """Validate Python version requirements."""
    issues = []
    
    current_version = sys.version_info[:2]
    if current_version < REQUIRED_PYTHON_VERSION:
        issues.append(f"Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}+ required, found {current_version[0]}.{current_version[1]}")
        issues.append(f"  Solution: Install Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]} or later")
    else:
        print_success(f"Python version {current_version[0]}.{current_version[1]} meets requirements")
    
    return len(issues) == 0, issues

def validate_system_commands() -> Tuple[bool, List[str]]:
    """Validate required system commands are available."""
    issues = []
    
    print_info("Checking system commands...")
    for command, description in REQUIRED_COMMANDS.items():
        success, output = run_command(f"{command} --version")
        if success:
            # Extract version info
            version = output.split('\n')[0] if output else "unknown version"
            print_success(f"{command}: {version}")
        else:
            issues.append(f"Missing command: {command} ({description})")
            
            # Provide installation suggestions
            if command == "node" or command == "npm":
                issues.append("  Solution: Install Node.js from https://nodejs.org/")
            elif command == "ollama":
                issues.append("  Solution: Install Ollama from https://ollama.ai/download")
            elif command == "chroma":
                issues.append("  Solution: pip install chromadb")
    
    return len(issues) == 0, issues

def validate_python_packages() -> Tuple[bool, List[str]]:
    """Validate required Python packages are installed."""
    issues = []
    
    print_info("Checking Python packages...")
    for package, description in REQUIRED_PYTHON_PACKAGES:
        try:
            spec = importlib.util.find_spec(package.replace("-", "_"))
            if spec is not None:
                print_success(f"{package}: installed")
            else:
                issues.append(f"Missing Python package: {package} ({description})")
        except ImportError:
            issues.append(f"Missing Python package: {package} ({description})")
    
    if issues:
        issues.append("  Solution: pip install -r requirements.txt")
    
    return len(issues) == 0, issues

def validate_project_structure() -> Tuple[bool, List[str]]:
    """Validate project directory structure."""
    issues = []
    
    print_info("Checking project structure...")
    
    # Check directories
    for directory in REQUIRED_DIRECTORIES:
        if Path(directory).is_dir():
            print_success(f"Directory: {directory}")
        else:
            issues.append(f"Missing directory: {directory}")
    
    # Check files
    for file_path in REQUIRED_FILES:
        if Path(file_path).is_file():
            print_success(f"File: {file_path}")
        else:
            issues.append(f"Missing file: {file_path}")
    
    if issues:
        issues.append("  Solution: Ensure you're running from the project root directory")
    
    return len(issues) == 0, issues

def validate_environment_variables() -> Tuple[bool, List[str]]:
    """Validate environment variables and configuration."""
    issues = []
    warnings = []
    
    print_info("Checking environment variables...")
    
    # Critical environment variables
    if not os.getenv("TAVILY_API_KEY"):
        warnings.append("TAVILY_API_KEY not set - web search functionality will not work")
        warnings.append("  Solution: export TAVILY_API_KEY='your_api_key'")
        warnings.append("  Get key from: https://tavily.com/")
    else:
        print_success("TAVILY_API_KEY is set")
    
    # Optional environment variables with defaults
    optional_vars = {
        "REACT_APP_CHROMA_URL": "http://localhost:8000",
        "REACT_APP_OLLAMA_URL": "http://localhost:11434", 
        "REACT_APP_RESEARCH_URL": "http://localhost:8001/research",
        "MCP_SERVER_URL": "http://localhost:8002/mcp"
    }
    
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        print_success(f"{var}: {value}")
    
    # Print warnings as info (not errors)
    for warning in warnings:
        if warning.startswith("  "):
            print_suggestion(warning[2:])
        else:
            print_warning(warning)
    
    return True, []  # Environment issues are warnings, not errors

def validate_ports() -> Tuple[bool, List[str]]:
    """Validate required ports are available."""
    issues = []
    
    print_info("Checking port availability...")
    
    for port, service in REQUIRED_PORTS.items():
        if check_port_available(port):
            print_success(f"Port {port} ({service}): available")
        else:
            issues.append(f"Port {port} ({service}) is already in use")
    
    if issues:
        issues.append("  Solution: Stop other services using these ports or change configuration")
    
    return len(issues) == 0, issues

def validate_node_dependencies() -> Tuple[bool, List[str]]:
    """Validate Node.js dependencies."""
    issues = []
    
    print_info("Checking Node.js dependencies...")
    
    package_json_path = Path("llm-rag-chat/package.json")
    node_modules_path = Path("llm-rag-chat/node_modules")
    
    if not package_json_path.exists():
        issues.append("package.json not found in llm-rag-chat/")
        return False, issues
    
    if not node_modules_path.exists():
        issues.append("Node.js dependencies not installed")
        issues.append("  Solution: cd llm-rag-chat && npm install")
    else:
        print_success("Node.js dependencies are installed")
    
    return len(issues) == 0, issues

def validate_ollama_models() -> Tuple[bool, List[str]]:
    """Validate Ollama models are available."""
    issues = []
    warnings = []
    
    print_info("Checking Ollama models...")
    
    required_models = ["llama3.1", "nomic-embed-text"]
    
    success, output = run_command("ollama list")
    if not success:
        warnings.append("Cannot check Ollama models - ensure Ollama is running")
        warnings.append("  Solution: ollama serve")
        return True, warnings
    
    available_models = output.lower()
    
    for model in required_models:
        if model.lower() in available_models:
            print_success(f"Ollama model: {model}")
        else:
            warnings.append(f"Ollama model not found: {model}")
            warnings.append(f"  Solution: ollama pull {model}")
    
    # Print warnings
    for warning in warnings:
        if warning.startswith("  "):
            print_suggestion(warning[2:])
        else:
            print_warning(warning)
    
    return True, []  # Model issues are warnings

def validate_data_files() -> Tuple[bool, List[str]]:
    """Validate required data files."""
    issues = []
    warnings = []
    
    print_info("Checking data files...")
    
    data_file = Path("data/cleaned_slang_data.csv")
    if data_file.exists():
        file_size = data_file.stat().st_size
        print_success(f"Data file: {data_file} ({file_size:,} bytes)")
    else:
        warnings.append("cleaned_slang_data.csv not found")
        warnings.append("  Note: RAG system may not work without training data")
        warnings.append("  Solution: Run the data preparation notebook")
    
    # Check ChromaDB
    chroma_dir = Path("chroma")
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        print_success("ChromaDB data directory exists")
    else:
        warnings.append("ChromaDB data not found") 
        warnings.append("  Note: Vector database needs to be initialized")
        warnings.append("  Solution: Run the RAG setup notebook")
    
    # Print warnings
    for warning in warnings:
        if warning.startswith("  "):
            print_suggestion(warning[2:])
        else:
            print_warning(warning)
    
    return True, []  # Data issues are warnings

def validate_internet_connection() -> Tuple[bool, List[str]]:
    """Validate internet connectivity."""
    issues = []
    
    print_info("Checking internet connection...")
    
    if check_internet_connection():
        print_success("Internet connection: available")
    else:
        issues.append("No internet connection detected")
        issues.append("  Note: Required for web search functionality")
        issues.append("  Solution: Check your network connection")
    
    return len(issues) == 0, issues

# ===================================================================
# MAIN VALIDATION ORCHESTRATOR
# ===================================================================

def run_all_validations() -> None:
    """Run all validation checks and provide comprehensive report."""
    
    print_header(f"🔍 ME344 Environment Validation Script v{SCRIPT_VERSION}")
    print_info("Checking system requirements and configuration...")
    
    validations = [
        ("Python Version", validate_python_version),
        ("System Commands", validate_system_commands), 
        ("Python Packages", validate_python_packages),
        ("Project Structure", validate_project_structure),
        ("Environment Variables", validate_environment_variables),
        ("Port Availability", validate_ports),
        ("Node.js Dependencies", validate_node_dependencies),
        ("Ollama Models", validate_ollama_models),
        ("Data Files", validate_data_files),
        ("Internet Connection", validate_internet_connection)
    ]
    
    results = []
    all_issues = []
    
    for name, validation_func in validations:
        print_header(f"🔍 {name}")
        try:
            success, issues = validation_func()
            results.append((name, success))
            
            if issues:
                all_issues.extend(issues)
                for issue in issues:
                    if issue.startswith("  "):
                        print_suggestion(issue[2:])
                    else:
                        print_error(issue)
        except Exception as e:
            print_error(f"Validation failed: {e}")
            results.append((name, False))
            all_issues.append(f"{name} validation failed: {e}")
    
    # Print summary
    print_header("📋 Validation Summary")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n{Colors.BOLD}Overall Status: {passed}/{total} checks passed{Colors.END}")
    
    for name, success in results:
        if success:
            print_success(f"{name}")
        else:
            print_error(f"{name}")
    
    if passed == total:
        print_header("🎉 Environment Ready!")
        print_success("All validation checks passed!")
        print_info("Your system is ready to run the ME344 Deep Research Agent.")
        print_info("Next steps:")
        print_suggestion("1. Run: ./run_part2.sh")
        print_suggestion("2. Open: http://localhost:3000")
    else:
        print_header("⚠️  Issues Found")
        print_warning(f"{total - passed} validation check(s) failed.")
        print_info("Please fix the issues above before running the system.")
        print_info("Most issues can be resolved by:")
        print_suggestion("1. Installing missing dependencies")
        print_suggestion("2. Setting required environment variables") 
        print_suggestion("3. Running from the correct directory")
        
        if any("TAVILY_API_KEY" in issue for issue in all_issues):
            print_info("\nTo get a Tavily API key:")
            print_suggestion("1. Visit: https://tavily.com/")
            print_suggestion("2. Sign up for an account")
            print_suggestion("3. Get your API key")
            print_suggestion("4. Export: export TAVILY_API_KEY='your_key'")

def main() -> None:
    """Main entry point."""
    try:
        run_all_validations()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()