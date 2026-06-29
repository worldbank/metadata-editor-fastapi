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
    echo "  --foreground, -f        Run in foreground (errors visible in terminal; Ctrl+C to stop)"
    echo "  --clear-jobs            Delete job store and result files before starting (run after ./stop.sh)"
    echo "  --python-version VER   Use specific Python version (e.g., 3.13, 3.12)"
    echo ""
    echo "Environment variables:"
    echo "  HOST             Server host (default: 127.0.0.1)"
    echo "  PORT             Server port (default: 8000)"
    echo "  PYTHON_VERSION   Specific Python version to use (e.g., 3.13, 3.12)"
    echo "  STORAGE_PATH     Required in .env — directory path or empty to disable validation"
    echo "  CONDA_ENV_NAME   Conda environment name to use (default: metadata-editor)"
    echo ""
    echo "Python Environment Detection (in priority order):"
    echo "  1. Conda environment named 'metadata-editor' (or \$CONDA_ENV_NAME)"
    echo "  2. Currently active conda environment (CONDA_DEFAULT_ENV is set)"
    echo "  3. Virtual environment (.venv/) if available"
    echo "  4. Specific Python version if --python-version or PYTHON_VERSION is set"
    echo "  5. Available system Python versions (3.13, 3.12, 3.11, etc.)"
    echo "  6. System uvicorn command directly"
    echo ""
    echo "Examples:"
    echo "  $0                                       # Start in background (default)"
    echo "  $0 --foreground                          # Start in foreground for debugging"
    echo "  ./stop.sh && $0 --clear-jobs             # Stop, clear queued/finished jobs, restart fresh"
    echo "  HOST=0.0.0.0 $0                         # Bind on all interfaces (advanced)"
    echo "  PORT=8000 $0                            # Start on port 8000"
    echo "  CONDA_ENV_NAME=myenv $0                 # Use a custom conda environment"
    echo "  $0 --python-version 3.13                # Use Python 3.13 specifically"
    echo ""
    echo "Quick Commands:"
    echo "  $0 --check                              # Run checks only"
    echo "  $0 --help                               # Show this help"
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
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="8000"

# Python version configuration
PYTHON_VERSION="${PYTHON_VERSION:-}"  # Allow override via environment variable

# Conda environment name (used when conda is the preferred environment manager)
CONDA_ENV_NAME="${CONDA_ENV_NAME:-metadata-editor}"

# Detect Python executable
PYTHON_EXEC=""
UVICORN_EXEC=""
ENV_SOURCE=""
FOREGROUND=false
CLEAR_JOBS=false

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

# Resolve JOB_STORE_DB_PATH (env, .env file, or default)
resolve_job_store_path() {
    local path="${JOB_STORE_DB_PATH:-}"
    if [ -z "$path" ] && [ -f "$PROJECT_DIR/.env" ]; then
        path=$(grep -E '^[[:space:]]*JOB_STORE_DB_PATH=' "$PROJECT_DIR/.env" | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'" | xargs)
    fi
    if [ -z "$path" ]; then
        path="db/jobs.sqlite"
    fi
    if [[ "$path" != /* ]]; then
        path="$PROJECT_DIR/$path"
    fi
    echo "$path"
}

# Delete durable job store and result JSON files (startup reset)
clear_jobs() {
    print_warning "Clearing job store and result files before startup..."

    local store_path
    store_path="$(resolve_job_store_path)"
    local removed_store=0
    local removed_results=0

    for f in "$store_path" "${store_path}-wal" "${store_path}-shm"; do
        if [ -f "$f" ]; then
            rm -f "$f"
            removed_store=$((removed_store + 1))
            print_status "Removed: $f"
        fi
    done

    if [ -d "$PROJECT_DIR/jobs" ]; then
        shopt -s nullglob
        local result_files=("$PROJECT_DIR/jobs"/*.json)
        shopt -u nullglob
        for f in "${result_files[@]}"; do
            rm -f "$f"
            removed_results=$((removed_results + 1))
        done
    fi

    print_success "Job clear complete (store files: $removed_store, result files: $removed_results)"
    print_status "Job audit log (logs/job_audit.log) was not cleared"
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
    print_status "Detecting Python environment..."

    # --- 1. Named conda environment ---
    if command -v conda > /dev/null 2>&1; then
        print_status "Conda found. Checking for environment: $CONDA_ENV_NAME"
        if conda env list 2>/dev/null | grep -qE "^${CONDA_ENV_NAME}[[:space:]]"; then
            local conda_python
            conda_python=$(conda run -n "$CONDA_ENV_NAME" python -c "import sys; print(sys.executable)" 2>/dev/null)
            if [ -n "$conda_python" ] && "$conda_python" -c "import uvicorn" 2>/dev/null; then
                PYTHON_EXEC="$conda_python"
                UVICORN_EXEC="$conda_python -m uvicorn"
                ENV_SOURCE="conda:$CONDA_ENV_NAME"
                print_success "Using conda environment '$CONDA_ENV_NAME': $PYTHON_EXEC"
                return 0
            else
                print_warning "Conda env '$CONDA_ENV_NAME' found but uvicorn is not installed in it"
                print_warning "Run: conda activate $CONDA_ENV_NAME && pip install -r requirements.txt"
            fi
        else
            print_warning "Conda environment '$CONDA_ENV_NAME' not found"
        fi

        # --- 2. Active conda environment ---
        if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
            print_status "Active conda environment detected: $CONDA_DEFAULT_ENV"
            local active_python
            active_python=$(python -c "import sys; print(sys.executable)" 2>/dev/null)
            if [ -n "$active_python" ] && "$active_python" -c "import uvicorn" 2>/dev/null; then
                PYTHON_EXEC="$active_python"
                UVICORN_EXEC="$active_python -m uvicorn"
                ENV_SOURCE="conda-active:$CONDA_DEFAULT_ENV"
                print_success "Using active conda environment '$CONDA_DEFAULT_ENV': $PYTHON_EXEC"
                return 0
            else
                print_warning "Active conda env '$CONDA_DEFAULT_ENV' found but uvicorn is not installed"
            fi
        fi
    fi

    # --- 3. Virtual environment (.venv) ---
    if check_venv_directory; then
        PYTHON_EXEC="$VENV_DIR/bin/python"
        UVICORN_EXEC="$VENV_DIR/bin/uvicorn"
        ENV_SOURCE="venv"
        print_success "Using virtual environment Python: $PYTHON_EXEC"
        return 0
    fi

    # --- 4. System Python versions ---
    local python_versions=($(find_python_versions))

    if [ ${#python_versions[@]} -eq 0 ]; then
        print_error "No Python executables found"
        return 1
    fi

    print_status "Found Python versions: ${python_versions[*]}"

    for python_cmd in "${python_versions[@]}"; do
        print_status "Testing $python_cmd..."
        if test_python_executable "$python_cmd"; then
            PYTHON_EXEC="$python_cmd"
            UVICORN_EXEC="$python_cmd -m uvicorn"
            ENV_SOURCE="system"
            print_success "Using system Python: $PYTHON_EXEC"
            local version_info
            version_info=$($python_cmd --version 2>&1)
            print_status "Python version: $version_info"
            return 0
        else
            print_warning "$python_cmd found but uvicorn not available"
        fi
    done

    # --- 5. Uvicorn directly as last resort ---
    if command -v uvicorn > /dev/null 2>&1; then
        PYTHON_EXEC="python3"
        UVICORN_EXEC="uvicorn"
        ENV_SOURCE="system-uvicorn"
        print_success "Using system uvicorn directly"
        return 0
    fi

    print_error "Could not find Python with uvicorn installed"
    print_error "Please set up an environment using one of these options:"
    print_error ""
    print_error "Option A - Conda (recommended for geospatial support):"
    print_error "  conda create -n metadata-editor python=3.11 -y"
    print_error "  conda activate metadata-editor"
    print_error "  conda install -c conda-forge gdal fiona geopandas rasterio pyproj shapely -y"
    print_error "  pip install -r requirements.txt"
    print_error ""
    print_error "Option B - Virtual environment:"
    print_error "  python3 -m venv .venv"
    print_error "  source .venv/bin/activate"
    print_error "  pip install -r requirements.txt"
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

# Resolve STORAGE_PATH from shell env or .env (returns __MISSING__ if absent)
resolve_storage_path_setting() {
    if [ -n "${STORAGE_PATH+set}" ]; then
        printf '%s' "$STORAGE_PATH"
        return 0
    fi
    if [ -f "$PROJECT_DIR/.env" ] && grep -qE '^[[:space:]]*STORAGE_PATH=' "$PROJECT_DIR/.env"; then
        grep -E '^[[:space:]]*STORAGE_PATH=' "$PROJECT_DIR/.env" | tail -1 | cut -d= -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | tr -d '"' | tr -d "'"
        return 0
    fi
    printf '%s' "__MISSING__"
}

# Function to check environment configuration
check_env_config() {
    print_status "Checking environment configuration..."
    
    # Check if .env file exists
    if [ -f "$PROJECT_DIR/.env" ]; then
        print_success "Found .env configuration file"
    else
        print_warning "No .env file found - copy .env.example to .env before starting"
    fi
    
    local storage_setting
    storage_setting="$(resolve_storage_path_setting)"
    if [ "$storage_setting" = "__MISSING__" ]; then
        print_error "STORAGE_PATH must be set in .env"
        print_error "Use an absolute directory path, or STORAGE_PATH= (empty) for local dev only"
        exit 1
    fi
    if [ -z "$storage_setting" ]; then
        print_warning "STORAGE_PATH is empty - path validation disabled (local development only)"
    elif [ ! -d "$storage_setting" ]; then
        print_error "STORAGE_PATH directory does not exist: $storage_setting"
        print_error "Please create the directory or update your .env file"
        exit 1
    else
        print_success "STORAGE_PATH is valid: $storage_setting"
    fi
}

# Function to start the application
start_app() {
    print_status "Starting Metadata Editor FastAPI application..."
    
    # Get host and port from environment or use defaults
    local host="${HOST:-$DEFAULT_HOST}"
    local port="${PORT:-$DEFAULT_PORT}"
    
    print_status "Configuration:"
    print_status "  Host:        $host"
    print_status "  Port:        $port"
    print_status "  Python:      $PYTHON_EXEC"
    print_status "  Environment: $ENV_SOURCE"
    print_status "  Mode:        $([ "$FOREGROUND" = true ] && echo foreground || echo background)"
    print_status "  Log file:    $LOG_FILE"

    cd "$PROJECT_DIR"

    if [ "$FOREGROUND" = true ]; then
        print_status "Starting in foreground (Ctrl+C to stop)..."
        print_status "  Application URL: http://$host:$port"
        print_status "  API Documentation: http://$host:$port/docs"
        exec $UVICORN_EXEC main:app \
            --host "$host" \
            --port "$port" \
            --log-level info
    fi

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
        --foreground|-f)
            FOREGROUND=true
            shift
            ;;
        --clear-jobs)
            CLEAR_JOBS=true
            shift
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

    if [ "$CLEAR_JOBS" = true ]; then
        clear_jobs
    fi

    # Start the application
    start_app
    
    print_success "=== Application startup completed ==="
}

main
