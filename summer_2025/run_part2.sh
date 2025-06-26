#!/bin/bash

# ===================================================================
# ME344 Deep Research Agent Startup Script
# Starts all required services for the multi-agent RAG system
# ===================================================================

set -e  # Exit on any error

# ANSI color codes for better output formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Service PIDs for cleanup
declare MCP_PID AGENT_PID CHROMA_PID OLLAMA_PID NPM_PID

# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "${PURPLE}📍 $1${NC}"
}

print_separator() {
    echo -e "${CYAN}================================================================${NC}"
}

# ===================================================================
# VALIDATION FUNCTIONS
# ===================================================================

check_working_directory() {
    log_step "Validating working directory..."
    
    if [[ ! -f "run_part2.sh" ]]; then
        log_error "Script must be run from the project root directory"
        log_info "Expected files: run_part2.sh, mcp_server/, deep_research_agent/, llm-rag-chat/"
        exit 1
    fi
    
    local missing_dirs=()
    [[ ! -d "mcp_server" ]] && missing_dirs+=("mcp_server/")
    [[ ! -d "deep_research_agent" ]] && missing_dirs+=("deep_research_agent/")
    [[ ! -d "llm-rag-chat" ]] && missing_dirs+=("llm-rag-chat/")
    
    if [[ ${#missing_dirs[@]} -gt 0 ]]; then
        log_error "Missing required directories: ${missing_dirs[*]}"
        log_info "Please ensure you're in the correct project directory"
        exit 1
    fi
    
    log_success "Working directory validated"
}

check_python_environment() {
    log_step "Checking Python virtual environment..."
    
    local venv_paths=("$HOME/codes/python/.venv" "./.venv" "./venv")
    local venv_found=false
    
    for venv_path in "${venv_paths[@]}"; do
        if [[ -d "$venv_path" && -f "$venv_path/bin/activate" ]]; then
            log_info "Found virtual environment: $venv_path"
            # shellcheck source=/dev/null
            source "$venv_path/bin/activate"
            venv_found=true
            break
        fi
    done
    
    if [[ "$venv_found" = false ]]; then
        log_error "No Python virtual environment found"
        log_info "Please create a virtual environment in one of these locations:"
        for path in "${venv_paths[@]}"; do
            echo -e "  ${CYAN}$path${NC}"
        done
        log_info ""
        log_info "To create a virtual environment:"
        log_info "  python3 -m venv ./.venv"
        log_info "  source ./.venv/bin/activate"
        log_info "  pip install -r requirements.txt"
        exit 1
    fi
    
    # Verify Python and pip are available
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found in virtual environment"
        log_info "Please ensure your virtual environment has Python3 installed"
        exit 1
    fi
    
    log_success "Virtual environment activated: $(which python3)"
}

check_python_dependencies() {
    log_step "Checking Python dependencies..."
    
    local required_packages=("fastapi" "uvicorn" "langchain" "chromadb" "httpx" "tavily")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import ${package//-/_}" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log_error "Missing required Python packages: ${missing_packages[*]}"
        log_info "Please install dependencies:"
        log_info "  pip install -r requirements.txt"
        exit 1
    fi
    
    log_success "Python dependencies verified"
}

check_environment_variables() {
    log_step "Checking environment variables..."
    
    local warnings=()
    
    if [[ -z "$TAVILY_API_KEY" ]]; then
        warnings+=("TAVILY_API_KEY not set - web search functionality will fail")
        log_info "To fix: export TAVILY_API_KEY='your_api_key_here'"
    fi
    
    # Set default environment variables if not set
    # Set flexible CORS configuration
    export FRONTEND_PORT="${FRONTEND_PORT:-3000}"
    local cors_origins="${CHROMA_SERVER_CORS_ALLOW_ORIGINS:-[\"http://localhost:${FRONTEND_PORT}\"]}"
    
    export CHROMA_SERVER_CORS_ALLOW_ORIGINS="${cors_origins}"
    
    log_info "CORS configuration: ${cors_origins}"
    log_info "Frontend will run on port: ${FRONTEND_PORT}"
    export REACT_APP_CHROMA_URL="${REACT_APP_CHROMA_URL:-http://localhost:8000}"
    export REACT_APP_OLLAMA_URL="${REACT_APP_OLLAMA_URL:-http://localhost:11434}"
    export REACT_APP_RESEARCH_URL="${REACT_APP_RESEARCH_URL:-http://localhost:8001/research}"
    # MCP_SERVER_URL removed - Deep Research Agent now uses Tavily directly
    export CHROMA_PERSIST_DIR="${CHROMA_PERSIST_DIR:-./deep_research_chroma}"
    export RESEARCH_COLLECTION_NAME="${RESEARCH_COLLECTION_NAME:-deep_research_collection}"
    export LLM_MODEL="${LLM_MODEL:-llama3.1}"
    export EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
    
    if [[ ${#warnings[@]} -gt 0 ]]; then
        log_warning "Environment variable warnings:"
        for warning in "${warnings[@]}"; do
            echo -e "  ${YELLOW}• $warning${NC}"
        done
        echo ""
        log_info "Continue? Some features may not work properly. (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Setup cancelled. Please set required environment variables."
            exit 1
        fi
    fi
    
    log_success "Environment variables configured"
}

check_system_requirements() {
    log_step "Checking system requirements..."
    
    local required_commands=("node" "npm" "ollama")
    local missing_commands=()
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_commands+=("$cmd")
        fi
    done
    
    # Check for chroma in venv
    if [[ ! -f "./.venv/bin/chroma" ]] && ! command -v chroma &> /dev/null; then
        missing_commands+=("chroma")
    fi
    
    if [[ ${#missing_commands[@]} -gt 0 ]]; then
        log_error "Missing required system commands: ${missing_commands[*]}"
        log_info ""
        log_info "Installation instructions:"
        for cmd in "${missing_commands[@]}"; do
            case "$cmd" in
                "node"|"npm")
                    log_info "  Node.js: https://nodejs.org/ or use nvm"
                    ;;
                "ollama")
                    log_info "  Ollama: https://ollama.ai/download"
                    ;;
                "chroma")
                    log_info "  ChromaDB: pip install chromadb"
                    ;;
            esac
        done
        exit 1
    fi
    
    log_success "System requirements verified"
}

check_node_dependencies() {
    log_step "Checking Node.js dependencies..."
    
    if [[ ! -d "llm-rag-chat/node_modules" ]]; then
        log_warning "Node.js dependencies not installed"
        log_info "Installing dependencies..."
        cd llm-rag-chat
        if ! npm install; then
            log_error "Failed to install Node.js dependencies"
            exit 1
        fi
        cd ..
    fi
    
    log_success "Node.js dependencies verified"
}

kill_processes_on_port() {
    local port="$1"
    local service_name="$2"
    
    if command -v lsof &> /dev/null; then
        local pids
        pids=$(lsof -ti ":$port" 2>/dev/null)
        if [[ -n "$pids" ]]; then
            log_info "Killing processes on port $port ($service_name)..."
            echo "$pids" | xargs kill -9 2>/dev/null || true
            sleep 1
            
            # Verify port is free
            if ! lsof -Pi ":$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
                log_success "Port $port freed"
            fi
        fi
    elif command -v netstat &> /dev/null && command -v kill &> /dev/null; then
        # Fallback method for systems without lsof
        log_warning "Using fallback method to kill processes on port $port"
        log_info "You may need to manually kill processes using: sudo lsof -ti :$port | xargs kill -9"
    fi
}

check_port_availability() {
    log_step "Checking port availability..."
    
    local ports=(${FRONTEND_PORT} 8000 8001 11434)
    local port_descriptions=("React Frontend" "ChromaDB" "Deep Research Agent" "Ollama")
    local busy_ports=()
    local busy_descriptions=()
    
    for i in "${!ports[@]}"; do
        local port="${ports[$i]}"
        local desc="${port_descriptions[$i]}"
        
        if command -v lsof &> /dev/null; then
            if lsof -Pi ":$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
                busy_ports+=("$port")
                busy_descriptions+=("$desc")
            fi
        elif command -v netstat &> /dev/null; then
            if netstat -ln 2>/dev/null | grep -q ":$port "; then
                busy_ports+=("$port")
                busy_descriptions+=("$desc")
            fi
        fi
    done
    
    if [[ ${#busy_ports[@]} -gt 0 ]]; then
        log_warning "The following ports are already in use:"
        for i in "${!busy_ports[@]}"; do
            echo -e "  ${YELLOW}Port ${busy_ports[$i]}${NC} - ${busy_descriptions[$i]}"
        done
        echo ""
        
        # Special handling for known services
        local has_chromadb=false
        local has_ollama=false
        for port in "${busy_ports[@]}"; do
            [[ "$port" == "8000" ]] && has_chromadb=true
            [[ "$port" == "11434" ]] && has_ollama=true
        done
        
        if [[ "$has_chromadb" == true ]] || [[ "$has_ollama" == true ]]; then
            log_info "Detected running services that can be reused."
        fi
        
        log_info "What would you like to do?"
        log_info "  1) Kill the processes and free the ports"
        log_info "  2) Continue anyway (some services may fail)"
        log_info "  3) Cancel setup"
        echo -n "Choice (1-3): "
        read -r choice
        
        case "$choice" in
            1)
                log_info "Killing existing processes on busy ports..."
                for i in "${!busy_ports[@]}"; do
                    local port=${busy_ports[$i]}
                    local desc=${busy_descriptions[$i]}
                    kill_processes_on_port "$port" "$desc"
                done
                
                # Re-check ports after killing processes
                local still_busy=()
                for port in "${busy_ports[@]}"; do
                    if command -v lsof &> /dev/null; then
                        if lsof -Pi ":$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
                            still_busy+=("$port")
                        fi
                    fi
                done
                
                if [[ ${#still_busy[@]} -gt 0 ]]; then
                    log_error "Failed to free ports: ${still_busy[*]}"
                    log_info "You may need to manually kill these processes"
                    exit 1
                fi
                
                log_success "All ports freed successfully"
                ;;
            2)
                log_warning "Continuing with busy ports. Some services may fail to start."
                ;;
            *)
                log_info "Setup cancelled."
                exit 0
                ;;
        esac
    else
        log_success "All required ports are available"
    fi
    
    log_success "Port availability checked"
}

# ===================================================================
# SERVICE MANAGEMENT FUNCTIONS
# ===================================================================

start_service() {
    local service_name="$1"
    local command="$2"
    local port="$3"
    local pid_var="$4"
    local wait_time="${5:-2}"
    
    log_step "Starting $service_name on port $port..."
    
    # Start service in background
    eval "$command" &
    local service_pid=$!
    
    # Store PID in the specified variable
    eval "$pid_var=$service_pid"
    
    # Wait for service to start
    sleep "$wait_time"
    
    # Verify service is still running
    if ! kill -0 "$service_pid" 2>/dev/null; then
        log_error "$service_name failed to start"
        log_info "Check the logs above for error details"
        exit 1
    fi
    
    log_success "$service_name started (PID: $service_pid)"
}

wait_for_service() {
    local service_name="$1"
    local url="$2"
    local max_attempts="${3:-30}"
    local delay="${4:-2}"
    
    log_info "Waiting for $service_name to be ready..."
    
    for ((i=1; i<=max_attempts; i++)); do
        if curl -s -f "$url" >/dev/null 2>&1; then
            log_success "$service_name is ready"
            return 0
        fi
        
        if ((i % 5 == 0)); then
            log_info "Still waiting for $service_name... (attempt $i/$max_attempts)"
        fi
        
        sleep "$delay"
    done
    
    log_warning "$service_name may not be fully ready, but continuing..."
    return 1
}

# ===================================================================
# CLEANUP FUNCTION
# ===================================================================

cleanup() {
    echo ""
    log_info "Shutdown signal received. Stopping all services..."
    
    local services=("NPM_PID:React Frontend" "OLLAMA_PID:Ollama Server" 
                   "CHROMA_PID:ChromaDB" "AGENT_PID:Deep Research Agent")
    
    for service in "${services[@]}"; do
        IFS=':' read -r pid_var service_name <<< "$service"
        local pid_value
        pid_value=$(eval echo "\$$pid_var")
        
        if [[ -n "$pid_value" ]] && [[ "$pid_value" != "existing" ]] && kill -0 "$pid_value" 2>/dev/null; then
            log_info "Stopping $service_name (PID: $pid_value)..."
            kill "$pid_value" 2>/dev/null || true
            # Give process time to shutdown gracefully
            sleep 1
            # Force kill if still running
            if kill -0 "$pid_value" 2>/dev/null; then
                kill -9 "$pid_value" 2>/dev/null || true
            fi
        elif [[ "$pid_value" == "existing" ]]; then
            log_info "Skipping $service_name (was already running)"
        fi
    done
    
    log_success "All services stopped"
    exit 0
}

# ===================================================================
# MAIN EXECUTION
# ===================================================================

main() {
    # Set up signal handling
    trap cleanup INT TERM
    
    # Print header
    print_separator
    echo -e "${CYAN}🚀 ME344 Deep Research Agent - Startup Script${NC}"
    echo -e "${CYAN}   Multi-Service RAG System with MCP Integration${NC}"
    print_separator
    echo ""
    
    # Run all validation checks
    check_working_directory
    check_system_requirements
    check_python_environment
    check_python_dependencies
    check_environment_variables
    check_node_dependencies
    check_port_availability
    
    echo ""
    log_success "All validation checks passed! Starting services..."
    echo ""
    
    # Start all services in order
    # MCP Server removed - Deep Research Agent now uses Tavily directly
    
    start_service "Deep Research Agent" \
        "(cd ./deep_research_agent && uvicorn main:app --reload --port 8001)" \
        "8001" "AGENT_PID" 3
    
    # Check if ChromaDB is already running
    if curl -s "http://localhost:8000/api/v1/heartbeat" >/dev/null 2>&1; then
        log_success "ChromaDB already running on port 8000"
        CHROMA_PID="existing"
    else
        # Use chroma from venv if available
        local chroma_cmd="chroma"
        if [[ -f "./.venv/bin/chroma" ]]; then
            chroma_cmd="./.venv/bin/chroma"
        fi
        start_service "ChromaDB" \
            "($chroma_cmd run --host localhost --port 8000 --path ./chroma > /dev/null 2>&1)" \
            "8000" "CHROMA_PID" 4
    fi
    
    # Check if Ollama is already running
    if curl -s "http://localhost:11434/api/tags" >/dev/null 2>&1; then
        log_success "Ollama already running on port 11434"
        OLLAMA_PID="existing"
    else
        start_service "Ollama Server" \
            "(ollama serve > /dev/null 2>&1)" \
            "11434" "OLLAMA_PID" 5
    fi
    
    # Display important information BEFORE starting React (which produces verbose output)
    echo ""
    print_separator
    log_success "🎉 All backend services started successfully!"
    print_separator
    echo ""
    
    # Display service information
    echo -e "${CYAN}📋 Service URLs:${NC}"
    echo -e "  ${GREEN}React Frontend:${NC}     http://localhost:${FRONTEND_PORT} (starting...)"
    echo -e "  ${GREEN}ChromaDB:${NC}          http://localhost:8000 ✅"
    echo -e "  ${GREEN}Research Agent:${NC}    http://localhost:8001 ✅"
    echo -e "  ${GREEN}Ollama API:${NC}        http://localhost:11434 ✅"
    echo ""
    
    echo -e "${CYAN}🔧 Environment Variables:${NC}"
    echo -e "  ${GREEN}TAVILY_API_KEY:${NC}    ${TAVILY_API_KEY:-(not set)}"
    echo -e "  ${GREEN}LLM_MODEL:${NC}         $LLM_MODEL"
    echo -e "  ${GREEN}CHROMA_URL:${NC}        $REACT_APP_CHROMA_URL"
    echo -e "  ${GREEN}OLLAMA_URL:${NC}        $REACT_APP_OLLAMA_URL"
    echo -e "  ${GREEN}CHROMA_DIR:${NC}        $CHROMA_PERSIST_DIR"
    echo ""
    
    echo -e "${YELLOW}📝 Next Steps:${NC}"
    echo -e "  1. Open ${GREEN}http://localhost:${FRONTEND_PORT}${NC} in your browser"
    echo -e "  2. Toggle between RAG (Slang) and MCP (Deep Research) modes"
    echo -e "  3. Press ${RED}Ctrl+C${NC} in this terminal to stop all services"
    echo ""
    
    print_separator
    log_info "Starting React Frontend (this will produce verbose output)..."
    print_separator
    echo ""
    
    # Start React Frontend with output redirected to a log file
    local npm_log="./npm_start.log"
    log_info "React output will be logged to: $npm_log"
    start_service "React Frontend" \
        "(cd ./llm-rag-chat && PORT=${FRONTEND_PORT} npm start > ../npm_start.log 2>&1)" \
        "${FRONTEND_PORT}" "NPM_PID" 5
    
    # Wait a bit for React to start
    sleep 3
    
    # Clear the screen to ensure our instructions are visible
    clear
    
    # Display the important information again after clearing
    print_separator
    echo -e "${CYAN}🚀 ME344 Deep Research Agent - All Services Running${NC}"
    print_separator
    echo ""
    
    echo -e "${CYAN}📋 Service URLs:${NC}"
    echo -e "  ${GREEN}React Frontend:${NC}     http://localhost:${FRONTEND_PORT} ✅"
    echo -e "  ${GREEN}ChromaDB:${NC}          http://localhost:8000 ✅"
    echo -e "  ${GREEN}Research Agent:${NC}    http://localhost:8001 ✅"
    echo -e "  ${GREEN}Ollama API:${NC}        http://localhost:11434 ✅"
    echo ""
    
    echo -e "${YELLOW}📝 Next Steps:${NC}"
    echo -e "  1. Open ${GREEN}http://localhost:${FRONTEND_PORT}${NC} in your browser"
    echo -e "  2. Toggle between RAG (Slang) and MCP (Deep Research) modes"
    echo -e "  3. Press ${RED}Ctrl+C${NC} in this terminal to stop all services"
    echo ""
    
    echo -e "${CYAN}📄 Logs:${NC}"
    echo -e "  React frontend logs: ${GREEN}./npm_start.log${NC}"
    echo -e "  Watch logs: ${GREEN}tail -f npm_start.log${NC}"
    echo ""
    
    print_separator
    log_info "System is ready! Press Ctrl+C to shutdown all services."
    print_separator
    
    # Wait for interrupt signal
    wait
}

# Execute main function
main "$@"