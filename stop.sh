#!/bin/bash

# Metadata Editor FastAPI - Stop Script
# This script stops the FastAPI application gracefully

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$PROJECT_DIR/logs/app.pid"
LOG_FILE="$PROJECT_DIR/logs/app.log"

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

# Function to check if the application is running
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

# Function to stop the application gracefully
stop_app() {
    local pid=$(cat "$PID_FILE")
    
    print_status "Stopping application (PID: $pid)..."
    
    # Try graceful shutdown first (SIGTERM)
    if kill -TERM "$pid" 2>/dev/null; then
        print_status "Sent SIGTERM signal to process $pid"
        
        # Wait for graceful shutdown (up to 30 seconds)
        local count=0
        while [ $count -lt 30 ]; do
            if ! ps -p "$pid" > /dev/null 2>&1; then
                print_success "Application stopped gracefully"
                rm -f "$PID_FILE"
                return 0
            fi
            sleep 1
            count=$((count + 1))
            echo -n "."
        done
        echo ""
        
        # If still running, force kill
        print_warning "Application didn't stop gracefully, forcing shutdown..."
        if kill -KILL "$pid" 2>/dev/null; then
            print_success "Application force-stopped"
            rm -f "$PID_FILE"
            return 0
        else
            print_error "Failed to force-stop the application"
            return 1
        fi
    else
        print_error "Failed to send stop signal to process $pid"
        return 1
    fi
}

# Function to stop all uvicorn processes (fallback method)
stop_all_uvicorn() {
    print_status "Attempting to stop all uvicorn processes..."
    
    # Find uvicorn processes running main.py
    local uvicorn_pids=$(pgrep -f "uvicorn.*main:app" || true)
    
    if [ -z "$uvicorn_pids" ]; then
        print_warning "No uvicorn processes found"
        return 1
    fi
    
    print_status "Found uvicorn processes: $uvicorn_pids"
    
    # Stop each process
    for pid in $uvicorn_pids; do
        print_status "Stopping process $pid..."
        if kill -TERM "$pid" 2>/dev/null; then
            # Wait for graceful shutdown
            local count=0
            while [ $count -lt 10 ]; do
                if ! ps -p "$pid" > /dev/null 2>&1; then
                    print_success "Process $pid stopped gracefully"
                    break
                fi
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                print_warning "Force-stopping process $pid"
                kill -KILL "$pid" 2>/dev/null || true
            fi
        fi
    done
    
    # Clean up PID file
    rm -f "$PID_FILE"
    return 0
}

# Function to check for any remaining processes
check_remaining_processes() {
    local remaining_pids=$(pgrep -f "uvicorn.*main:app" || true)
    
    if [ -n "$remaining_pids" ]; then
        print_warning "Some uvicorn processes may still be running: $remaining_pids"
        print_warning "You may need to stop them manually if they're not part of this application"
        return 1
    fi
    
    return 0
}

# Main execution
main() {
    print_status "=== Metadata Editor FastAPI - Stop Script ==="
    print_status "Project directory: $PROJECT_DIR"
    
    # Check if application is running
    if ! is_app_running; then
        print_warning "Application is not running"
        
        # Check for any uvicorn processes anyway
        local uvicorn_pids=$(pgrep -f "uvicorn.*main:app" || true)
        if [ -n "$uvicorn_pids" ]; then
            print_warning "Found uvicorn processes that may be related: $uvicorn_pids"
            print_status "Use --force to stop all uvicorn processes"
        fi
        
        exit 0
    fi
    
    # Stop the application
    if stop_app; then
        print_success "Application stopped successfully"
    else
        print_error "Failed to stop application using PID file"
        
        # Try fallback method
        if [ "${1:-}" = "--force" ]; then
            print_status "Trying fallback method..."
            if stop_all_uvicorn; then
                print_success "Application stopped using fallback method"
            else
                print_error "Failed to stop application"
                exit 1
            fi
        else
            print_error "Use --force to try stopping all uvicorn processes"
            exit 1
        fi
    fi
    
    # Final check
    check_remaining_processes
    
    print_success "=== Application shutdown completed ==="
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Metadata Editor FastAPI - Stop Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h    Show this help message"
        echo "  --force       Force stop all uvicorn processes (use if PID method fails)"
        echo "  --status      Check if application is running"
        echo ""
        echo "Examples:"
        echo "  $0           # Stop application gracefully"
        echo "  $0 --force  # Force stop all uvicorn processes"
        echo "  $0 --status # Check application status"
        exit 0
        ;;
    --force)
        print_status "=== Force stopping all uvicorn processes ==="
        stop_all_uvicorn
        check_remaining_processes
        exit 0
        ;;
    --status)
        print_status "=== Checking application status ==="
        if is_app_running; then
            local pid=$(cat "$PID_FILE")
            print_success "Application is running (PID: $pid)"
            
            # Show process details
            if command -v ps > /dev/null; then
                print_status "Process details:"
                ps -p "$pid" -o pid,ppid,cmd,etime 2>/dev/null || true
            fi
            
            # Show port usage
            local port=$(lsof -p "$pid" 2>/dev/null | grep LISTEN | awk '{print $9}' | cut -d: -f2 | head -1)
            if [ -n "$port" ]; then
                print_status "Listening on port: $port"
            fi
        else
            print_warning "Application is not running"
        fi
        exit 0
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        print_error "Use --help for usage information"
        exit 1
        ;;
esac
