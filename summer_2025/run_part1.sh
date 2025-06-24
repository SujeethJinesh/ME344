#!/bin/bash

# ===================================================================
# ME344 RAG System Startup Script - Part 1
# Starts basic RAG system with ChromaDB and React frontend
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
declare CHROMA_PID OLLAMA_PID NPM_PID JUPYTER_PID

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
    
    if [[ ! -f "rag.ipynb" ]]; then
        log_error "Script must be run from the project root directory"
        log_info "Expected files: rag.ipynb, llm-rag-chat/, data/"
        exit 1
    fi
    
    local missing_dirs=()
    [[ ! -d "llm-rag-chat" ]] && missing_dirs+=("llm-rag-chat/")
    [[ ! -d "data" ]] && missing_dirs+=("data/")
    
    if [[ ${#missing_dirs[@]} -gt 0 ]]; then
        log_error "Missing required directories: ${missing_dirs[*]}"
        log_info "Please ensure you're in the correct project directory"
        exit 1
    fi
    
    log_success "Working directory validated"
}

check_python_environment() {
    log_step "Checking Python virtual environment..."
    
    local venv_paths=("$HOME/codes/python/python-venv" "./.venv" "./venv")
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
        log_info "  mkdir -p ~/codes/python && cd ~/codes/python"
        log_info "  python3.11 -m venv python-venv"
        log_info "  source ~/codes/python/python-venv/bin/activate"
        log_info "  pip install -r requirements.txt"
        exit 1
    fi
    
    # Verify Python and pip are available
    if ! command -v python &> /dev/null; then
        log_error "Python not found in virtual environment"
        log_info "Please ensure your virtual environment has Python installed"
        exit 1
    fi
    
    log_success "Virtual environment activated: $(which python)"
}

check_python_dependencies() {
    log_step "Checking Python dependencies..."
    
    local required_packages=("notebook" "langchain" "langchain_community" "chromadb" "langchain_chroma")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python -c "import ${package//-/_}" &> /dev/null; then
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

check_system_requirements() {
    log_step "Checking system requirements..."
    
    local required_commands=("node" "npm" "ollama" "chroma")
    local missing_commands=()
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_commands+=("$cmd")
        fi
    done
    
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
                    log_info "  Ollama: curl -fsSL https://ollama.com/install.sh | sh"
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

check_data_files() {
    log_step "Checking data files..."
    
    local data_file="data/cleaned_slang_data.csv"
    if [[ -f "$data_file" ]]; then
        local file_size=$(stat -c%s "$data_file" 2>/dev/null || stat -f%z "$data_file" 2>/dev/null || echo "unknown")
        log_success "Data file found: $data_file ($file_size bytes)"
    else {
        log_warning "Data file not found: $data_file"
        log_info "RAG system will work but may have limited knowledge"
        log_info "Consider adding your own dataset or downloading the Urban Dictionary dataset"
    }
    fi
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

check_port_availability() {
    log_step "Checking port availability..."
    
    local frontend_port="${FRONTEND_PORT:-3000}"
    local ports=(${frontend_port} 8000 8888 11434)
    local port_descriptions=("React Frontend" "ChromaDB" "Jupyter Notebook" "Ollama")
    local busy_ports=()
    
    for i in "${!ports[@]}"; do
        local port=${ports[$i]}
        local desc=${port_descriptions[$i]}
        
        if command -v lsof &> /dev/null; then
            if lsof -Pi ":$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
                busy_ports+=("$port ($desc)")
            fi
        elif command -v netstat &> /dev/null; then
            if netstat -ln 2>/dev/null | grep -q ":$port "; then
                busy_ports+=("$port ($desc)")
            fi
        fi
    done
    
    if [[ ${#busy_ports[@]} -gt 0 ]]; then
        log_warning "Ports already in use: ${busy_ports[*]}"
        log_info "These ports are required:"
        log_info "  ${frontend_port} - React Frontend"
        log_info "  8000 - ChromaDB"
        log_info "  8888 - Jupyter Notebook"
        log_info "  11434 - Ollama"
        log_info ""
        log_info "Continue anyway? Services may fail to start. (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Setup cancelled. Please free the required ports."
            exit 1
        fi
    fi
    
    log_success "Port availability checked"
}

check_ollama_models() {
    log_step "Checking Ollama models..."
    
    local required_models=("llama3.1" "nomic-embed-text")
    local missing_models=()
    
    # Check if Ollama is running first
    if ! pgrep -f "ollama serve" > /dev/null; then
        log_info "Ollama server not running, starting it..."
        ollama serve &
        OLLAMA_PID=$!
        sleep 3
    fi
    
    for model in "${required_models[@]}"; do
        if ! ollama list | grep -q "$model"; then
            missing_models+=("$model")
        fi
    done
    
    if [[ ${#missing_models[@]} -gt 0 ]]; then
        log_warning "Missing Ollama models: ${missing_models[*]}"
        log_info "Downloading missing models..."
        for model in "${missing_models[@]}"; do
            log_info "Pulling $model..."
            if ! ollama pull "$model"; then
                log_error "Failed to download $model"
                exit 1
            fi
        done
    fi
    
    log_success "Ollama models verified"
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

# ===================================================================
# CLEANUP FUNCTION
# ===================================================================

cleanup() {
    echo ""
    log_info "Shutdown signal received. Stopping all services..."
    
    local services=("NPM_PID:React Frontend" "OLLAMA_PID:Ollama Server" 
                   "CHROMA_PID:ChromaDB" "JUPYTER_PID:Jupyter Notebook")
    
    for service in "${services[@]}"; do
        IFS=':' read -r pid_var service_name <<< "$service"
        local pid_value
        pid_value=$(eval echo "\$$pid_var")
        
        if [[ -n "$pid_value" ]] && kill -0 "$pid_value" 2>/dev/null; then
            log_info "Stopping $service_name (PID: $pid_value)..."
            kill "$pid_value" 2>/dev/null || true
            # Give process time to shutdown gracefully
            sleep 1
            # Force kill if still running
            if kill -0 "$pid_value" 2>/dev/null; then
                kill -9 "$pid_value" 2>/dev/null || true
            fi
        fi
    done
    
    log_success "All services stopped"
    exit 0
}

# ===================================================================
# MAIN EXECUTION
# ===================================================================

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --notebook-only    Start only Jupyter notebook (for RAG development)"
    echo "  --frontend-only    Start only React frontend and required services"
    echo "  --help            Show this help message"
    echo ""
    echo "Default: Start all services for complete RAG system"
}

main() {
    # Parse command line arguments
    local notebook_only=false
    local frontend_only=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --notebook-only)
                notebook_only=true
                shift
                ;;
            --frontend-only)
                frontend_only=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Set up signal handling
    trap cleanup INT TERM
    
    # Print header
    print_separator
    echo -e "${CYAN}🚀 ME344 RAG System - Part 1 Startup Script${NC}"
    echo -e "${CYAN}   Basic RAG with ChromaDB and React Frontend${NC}"
    print_separator
    echo ""
    
    # Run validation checks
    check_working_directory
    check_system_requirements
    check_python_environment
    check_python_dependencies
    check_data_files
    check_node_dependencies
    check_port_availability
    check_ollama_models
    
    echo ""
    log_success "All validation checks passed! Starting services..."
    echo ""
    
    # Set environment variables with flexible CORS configuration
    local frontend_port="${FRONTEND_PORT:-3000}"
    local cors_origins="${CHROMA_SERVER_CORS_ALLOW_ORIGINS:-[\"http://localhost:${frontend_port}\"]}"
    
    export CHROMA_SERVER_CORS_ALLOW_ORIGINS="${cors_origins}"
    export DOCUMENTS_TO_ADD="${DOCUMENTS_TO_ADD:-500}"
    export FRONTEND_PORT="${frontend_port}"
    
    log_info "CORS configuration: ${cors_origins}"
    log_info "Frontend will run on port: ${frontend_port}"
    
    if [[ "$notebook_only" == true ]]; then
        # Start only Jupyter notebook for RAG development
        log_info "🔬 Starting in NOTEBOOK-ONLY mode"
        
        start_service "Jupyter Notebook" \
            "jupyter-notebook --no-browser --notebook-dir=$PWD" \
            "8888" "JUPYTER_PID" 3
            
    elif [[ "$frontend_only" == true ]]; then
        # Start only frontend services (assumes RAG is already set up)
        log_info "🌐 Starting in FRONTEND-ONLY mode"
        
        start_service "ChromaDB" \
            "(chroma run --host localhost --port 8000 --path ./chroma > /dev/null 2>&1)" \
            "8000" "CHROMA_PID" 4
        
        start_service "Ollama Server" \
            "(ollama serve > /dev/null 2>&1)" \
            "11434" "OLLAMA_PID" 5
        
        start_service "React Frontend" \
            "(cd ./llm-rag-chat && PORT=${frontend_port} npm start)" \
            "${frontend_port}" "NPM_PID" 3
    else
        # Start all services for complete RAG system
        log_info "🎯 Starting COMPLETE RAG system"
        
        start_service "ChromaDB" \
            "(chroma run --host localhost --port 8000 --path ./chroma > /dev/null 2>&1)" \
            "8000" "CHROMA_PID" 4
        
        start_service "Ollama Server" \
            "(ollama serve > /dev/null 2>&1)" \
            "11434" "OLLAMA_PID" 5
        
        start_service "React Frontend" \
            "(cd ./llm-rag-chat && PORT=${frontend_port} npm start)" \
            "${frontend_port}" "NPM_PID" 3
        
        start_service "Jupyter Notebook" \
            "jupyter-notebook --no-browser --notebook-dir=$PWD" \
            "8888" "JUPYTER_PID" 3
    fi
    
    echo ""
    print_separator
    log_success "🎉 Part 1 RAG System started successfully!"
    print_separator
    echo ""
    
    # Display service information
    echo -e "${CYAN}📋 Service URLs:${NC}"
    if [[ "$notebook_only" != true ]]; then
        echo -e "  ${GREEN}React Frontend:${NC}     http://localhost:${frontend_port}"
        echo -e "  ${GREEN}ChromaDB:${NC}          http://localhost:8000"
        echo -e "  ${GREEN}Ollama API:${NC}        http://localhost:11434"
    fi
    if [[ "$frontend_only" != true ]]; then
        echo -e "  ${GREEN}Jupyter Notebook:${NC}  http://localhost:8888"
    fi
    echo ""
    
    echo -e "${CYAN}🔧 Environment:${NC}"
    echo -e "  ${GREEN}Documents to add:${NC}   $DOCUMENTS_TO_ADD"
    echo -e "  ${GREEN}Data file:${NC}          data/cleaned_slang_data.csv"
    echo ""
    
    echo -e "${YELLOW}📝 Next Steps:${NC}"
    if [[ "$notebook_only" == true ]]; then
        echo -e "  1. Open ${GREEN}http://localhost:8888${NC} to access Jupyter"
        echo -e "  2. Run the ${GREEN}rag.ipynb${NC} notebook to set up your vector database"
        echo -e "  3. Once complete, restart with ${GREEN}--frontend-only${NC} to test your RAG system"
    elif [[ "$frontend_only" == true ]]; then
        echo -e "  1. Open ${GREEN}http://localhost:${frontend_port}${NC} to test your RAG system"
        echo -e "  2. Try asking questions about slang terms"
        echo -e "  3. Experiment with system prompt engineering in ${GREEN}Rag.js${NC}"
    else
        echo -e "  1. Open ${GREEN}http://localhost:8888${NC} to run the RAG notebook"
        echo -e "  2. Follow the notebook to populate your vector database"
        echo -e "  3. Open ${GREEN}http://localhost:${frontend_port}${NC} to test your RAG system"
    fi
    
    if [[ "$notebook_only" != true ]]; then
        echo ""
        echo -e "${BLUE}🌐 Port Forwarding (if on remote server):${NC}"
        echo -e "  ${CYAN}ssh -L ${frontend_port}:localhost:${frontend_port} -L 8000:localhost:8000 -L 8888:localhost:8888 -L 11434:localhost:11434 user@server${NC}"
    fi
    
    echo -e "  4. Press ${RED}Ctrl+C${NC} in this terminal to stop all services"
    echo ""
    
    print_separator
    log_info "System is ready! Press Ctrl+C to shutdown all services."
    print_separator
    
    # Wait for interrupt signal
    wait
}

# Execute main function
main "$@"