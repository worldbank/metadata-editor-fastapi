#!/bin/bash

# Metadata Editor FastAPI - Start Script
# This script starts the FastAPI application with proper environment setup

set -e  # Exit on any error

# Function to show help
show_help() {
    echo "Metadata Editor FastAPI - Start Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h              Show this help message"
    echo "  --check                 Only run checks without starting the application"
    echo "  --python-version VER   Use specific Python version (e.g., 3.13, 3.12)"
    echo ""
    echo "Environment variables:"
    echo "  HOST          Server host (default: 0.0.0.0)"
    echo "  PORT          Server port (default: 8000)"
    echo "  PYTHON_VERSION Specific Python version to use (e.g., 3.13, 3.12)"
    echo "  STORAGE_PATH  Path to data storage directory"
    echo ""
    echo "Python Environment Detection:"
    echo "  The script will automatically detect and use:"
    echo "  1. Virtual environment (.venv/) if available"
    echo "  2. Specific Python version if --python-version is specified"
    echo "  3. Specific Python version if PYTHON_VERSION environment variable is set"
    echo "  4. Available system Python versions (3.13, 3.12, 3.11, etc.)"
    echo "  5. System uvicorn command directly"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Start with default settings"
    echo "  HOST=127.0.0.1 $0                    # Start on localhost only"
    echo "  PORT=8000 $0                         # Start on port 8000"
    echo "  $0 --python-version 3.13             # Use Python 3.13 specifically"
    echo "  PYTHON_VERSION=3.13 $0               # Use Python 3.13 via environment"
    echo ""
    echo "Quick Commands:"
    echo "  $0 --check                           # Run checks only"
    echo "  $0 --help                            # Show this help"
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
MAIN_FILE="$PROJECT_DIR/main.py"
PID_FILE="$PROJECT_DIR/logs/app.pid"
LOG_FILE="$PROJECT_DIR/logs/app.log"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"

# Python version configuration
PYTHON_VERSION="${PYTHON_VERSION:-}"  # Allow override via environment variable

# Detect Python executable
PYTHON_EXEC=""
UVICORN_EXEC=""

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if the application is already running
is_app_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # Running
        else
            rm -f "$PID_FILE"  # Remove stale PID file
            return 1  # Not running
        fi
    fi
    return 1  # Not running
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    # Create logs directory if it doesn't exist
    if [ ! -d "$PROJECT_DIR/logs" ]; then
        mkdir -p "$PROJECT_DIR/logs"
        print_success "Created logs directory"
    fi
    
    # Create jobs directory if it doesn't exist
    if [ ! -d "$PROJECT_DIR/jobs" ]; then
        mkdir -p "$PROJECT_DIR/jobs"
        print_success "Created jobs directory"
    fi
}

# Function to check virtual environment directory
check_venv_directory() {
    if [ ! -d "$VENV_DIR" ]; then
        print_warning "Virtual environment directory not found: $VENV_DIR"
        print_warning "The script will attempt to use system Python instead"
        print_warning "To create a virtual environment, run:"
        print_warning "  python3 -m venv $VENV_DIR"
        print_warning "  source $VENV_DIR/bin/activate"
        print_warning "  pip install -r requirements.txt"
        return 1
    fi
    
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        print_warning "Python executable not found in virtual environment: $VENV_DIR/bin/python"
        print_warning "The virtual environment may be corrupted or incomplete"
        return 1
    fi
    
    return 0
}

# Function to find available Python versions
find_python_versions() {
    local versions=()
    
    # Check for specific version if requested
    if [ -n "$PYTHON_VERSION" ]; then
        # Try different formats for the requested version
        local found_version=""
        for format in "python$PYTHON_VERSION" "python3.$PYTHON_VERSION" "python$PYTHON_VERSION"; do
            if command -v "$format" > /dev/null 2>&1; then
                found_version="$format"
                break
            fi
        done
        
        if [ -n "$found_version" ]; then
            versions+=("$found_version")
            print_status "Found requested Python version: $found_version"
        else
            print_warning "Requested Python version $PYTHON_VERSION not found" >&2
            print_warning "Available formats checked: python$PYTHON_VERSION, python3.$PYTHON_VERSION" >&2
        fi
    fi
    
    # Find all available Python versions
    for version in python3.13 python3.12 python3.11 python3.10 python3.9 python3.8 python3 python; do
        if command -v "$version" > /dev/null 2>&1; then
            # Avoid duplicates
            if [[ ! " ${versions[@]} " =~ " ${version} " ]]; then
                versions+=("$version")
            fi
        fi
    done
    
    echo "${versions[@]}"
}

# Function to test Python executable with uvicorn
test_python_executable() {
    local python_cmd="$1"
    if $python_cmd -c "import uvicorn" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Function to detect Python and uvicorn executables
detect_python_executables() {
    # Check virtual environment directory first
    if check_venv_directory; then
        PYTHON_EXEC="$VENV_DIR/bin/python"
        UVICORN_EXEC="$VENV_DIR/bin/uvicorn"
        print_success "Using virtual environment Python: $PYTHON_EXEC"
        return 0
    fi
    
    # Get available Python versions
    local python_versions=($(find_python_versions))
    
    if [ ${#python_versions[@]} -eq 0 ]; then
        print_error "No Python executables found"
        return 1
    fi
    
    print_status "Found Python versions: ${python_versions[*]}"
    
    # Try each Python version
    for python_cmd in "${python_versions[@]}"; do
        print_status "Testing $python_cmd..."
        if test_python_executable "$python_cmd"; then
            PYTHON_EXEC="$python_cmd"
            UVICORN_EXEC="$python_cmd -m uvicorn"
            print_success "Using system Python: $PYTHON_EXEC"
            
            # Show Python version info
            local version_info=$($python_cmd --version 2>&1)
            print_status "Python version: $version_info"
            return 0
        else
            print_warning "$python_cmd found but uvicorn not available"
        fi
    done
    
    # Try uvicorn directly as last resort
    if command -v uvicorn > /dev/null 2>&1; then
        PYTHON_EXEC="python3"  # fallback
        UVICORN_EXEC="uvicorn"
        print_success "Using system uvicorn directly"
        return 0
    fi
    
    print_error "Could not find Python with uvicorn installed"
    print_error "Please either:"
    print_error "1. Create a virtual environment:"
    print_error "   python3 -m venv .venv"
    print_error "   source .venv/bin/activate"
    print_error "   pip install -r requirements.txt"
    print_error ""
    print_error "2. Install uvicorn globally:"
    print_error "   pip3 install uvicorn fastapi"
    print_error "   # or install all requirements:"
    print_error "   pip3 install -r requirements.txt"
    exit 1
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if uvicorn is available with detected Python
    if ! $PYTHON_EXEC -c "import uvicorn" 2>/dev/null; then
        print_error "uvicorn not found with $PYTHON_EXEC"
        print_error "Please install dependencies:"
        if [ -d "$VENV_DIR" ]; then
            print_error "  source .venv/bin/activate"
            print_error "  pip install -r requirements.txt"
        else
            print_error "  pip3 install -r requirements.txt"
        fi
        exit 1
    fi
    
    # Check if main.py exists
    if [ ! -f "$MAIN_FILE" ]; then
        print_error "Main application file not found: $MAIN_FILE"
        exit 1
    fi
    
    print_success "All dependencies are available"
}

# Function to check environment configuration
check_env_config() {
    print_status "Checking environment configuration..."
    
    # Check if .env file exists
    if [ -f "$PROJECT_DIR/.env" ]; then
        print_success "Found .env configuration file"
    else
        print_warning "No .env file found - using default configuration"
        print_warning "You can create a .env file with custom settings"
    fi
    
    # Check STORAGE_PATH if set
    if [ -n "$STORAGE_PATH" ]; then
        if [ ! -d "$STORAGE_PATH" ]; then
            print_error "STORAGE_PATH directory does not exist: $STORAGE_PATH"
            print_error "Please create the directory or update your .env file"
            exit 1
        fi
        print_success "STORAGE_PATH is valid: $STORAGE_PATH"
    else
        print_warning "STORAGE_PATH not set - path validation is disabled"
    fi
}

# Function to start the application
start_app() {
    print_status "Starting Metadata Editor FastAPI application..."
    
    # Get host and port from environment or use defaults
    local host="${HOST:-$DEFAULT_HOST}"
    local port="${PORT:-$DEFAULT_PORT}"
    
    print_status "Configuration:"
    print_status "  Host: $host"
    print_status "  Port: $port"
    print_status "  Python: $PYTHON_EXEC"
    print_status "  Log file: $LOG_FILE"
    
    # Start the application in the background
    nohup $UVICORN_EXEC main:app \
        --host "$host" \
        --port "$port" \
        --log-level info \
        > "$LOG_FILE" 2>&1 &
    
    local app_pid=$!
    
    # Write PID file with error handling
    if echo $app_pid > "$PID_FILE" 2>/dev/null; then
        print_success "PID file created: $PID_FILE"
    else
        print_error "Failed to create PID file: $PID_FILE"
        print_error "You may need to stop the application manually later"
        print_error "Process PID: $app_pid"
    fi
    
    # Wait a moment and check if the process is still running
    sleep 2
    
    if ps -p $app_pid > /dev/null 2>&1; then
        print_success "Application started successfully!"
        print_success "PID: $app_pid"
        print_success "Application URL: http://$host:$port"
        print_success "API Documentation: http://$host:$port/docs"
        print_success "Logs are being written to: $LOG_FILE"
        print_success ""
        print_status "To stop the application, run: ./stop.sh"
        print_status "To view logs in real-time: tail -f $LOG_FILE"
    else
        print_error "Failed to start the application"
        print_error "Check the log file for details: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --python-version=*)
            PYTHON_VERSION="${1#*=}"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --check)
            print_status "=== Running checks only ==="
            create_directories
            detect_python_executables
            check_dependencies
            check_env_config
            print_success "All checks passed!"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            print_error "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "=== Metadata Editor FastAPI - Start Script ==="
    print_status "Project directory: $PROJECT_DIR"
    
    # Check if already running
    if is_app_running; then
        local pid=$(cat "$PID_FILE")
        print_warning "Application is already running (PID: $pid)"
        print_warning "Use './stop.sh' to stop it first, or check './status.sh' for details"
        exit 1
    fi
    
    # Run checks
    create_directories
    detect_python_executables
    check_dependencies
    check_env_config
    
    # Start the application
    start_app
    
    print_success "=== Application startup completed ==="
}

# If we get here, no special options were processed
# Check if any arguments were provided
if [ $# -eq 0 ]; then
    # No arguments provided, show help
    show_help
    echo ""
    echo "Starting application with default settings..."
    echo ""
    main
else
    # Arguments were provided but not recognized
    print_error "Unknown option: $1"
    print_error "Use --help for usage information"
    exit 1
fi
