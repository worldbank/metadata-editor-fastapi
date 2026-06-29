#!/bin/bash

# Metadata Editor FastAPI - Application Control Script
# This script can start, stop, restart, and check the status of the FastAPI application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="$PROJECT_DIR/start.sh"
STOP_SCRIPT="$PROJECT_DIR/stop.sh"

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

# Function to check if scripts exist
check_scripts() {
    if [ ! -f "$START_SCRIPT" ]; then
        print_error "Start script not found: $START_SCRIPT"
        exit 1
    fi
    
    if [ ! -f "$STOP_SCRIPT" ]; then
        print_error "Stop script not found: $STOP_SCRIPT"
        exit 1
    fi
    
    # Make sure scripts are executable
    chmod +x "$START_SCRIPT" "$STOP_SCRIPT" 2>/dev/null || true
}

# Function to start the application
start_app() {
    print_status "Starting the application..."
    "$START_SCRIPT" "$@"
}

# Function to stop the application
stop_app() {
    print_status "Stopping the application..."
    "$STOP_SCRIPT" "$@"
}

# Function to restart the application
restart_app() {
    print_status "Restarting the application..."
    
    # Stop the application
    if "$STOP_SCRIPT" --status > /dev/null 2>&1; then
        print_status "Stopping current instance..."
        "$STOP_SCRIPT"
        sleep 2
    fi
    
    # Start the application
    print_status "Starting new instance..."
    "$START_SCRIPT" "$@"
}

# Function to show application status
show_status() {
    print_status "Checking application status..."
    "$STOP_SCRIPT" --status
}

# Function to show logs
show_logs() {
    local log_file="$PROJECT_DIR/logs/app.log"
    local lines="${1:-50}"
    
    if [ ! -f "$log_file" ]; then
        print_error "Log file not found: $log_file"
        return 1
    fi
    
    print_status "Showing last $lines lines of log file:"
    print_status "Log file: $log_file"
    echo "----------------------------------------"
    tail -n "$lines" "$log_file"
}

# Function to follow logs in real-time
follow_logs() {
    local log_file="$PROJECT_DIR/logs/app.log"
    
    if [ ! -f "$log_file" ]; then
        print_error "Log file not found: $log_file"
        return 1
    fi
    
    print_status "Following log file in real-time (Ctrl+C to stop):"
    print_status "Log file: $log_file"
    echo "----------------------------------------"
    tail -f "$log_file"
}

# Function to show help
show_help() {
    echo "Metadata Editor FastAPI - Application Control Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start       Start the application"
    echo "  stop        Stop the application"
    echo "  restart     Restart the application"
    echo "  status      Show application status"
    echo "  logs        Show recent log entries (default: 50 lines)"
    echo "  follow      Follow logs in real-time"
    echo "  help        Show this help message"
    echo ""
    echo "Start/Stop Options:"
    echo "  --help, -h    Show help for the specific command"
    echo "  --force       Force operation (for stop command)"
    echo "  --check       Only run checks without starting (for start command)"
    echo ""
    echo "Log Options:"
    echo "  logs N        Show last N lines of logs (default: 50)"
    echo ""
        echo "Environment Variables (for start/restart):"
        echo "  HOST          Server host (default: 127.0.0.1)"
        echo "  PORT          Server port (default: 8000)"
        echo "  RELOAD        Enable auto-reload (default: true)"
        echo "  STORAGE_PATH  Path to data storage directory"
        echo ""
        echo "Python Environment:"
        echo "  The scripts automatically detect and use:"
        echo "  1. Virtual environment (venv313/) if available"
        echo "  2. System Python3 with uvicorn installed"
        echo "  3. System Python with uvicorn installed"
        echo "  4. System uvicorn command directly"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start the application"
    echo "  $0 stop                     # Stop the application"
    echo "  $0 restart                  # Restart the application"
    echo "  $0 status                   # Check application status"
    echo "  $0 logs 100                 # Show last 100 log lines"
    echo "  $0 follow                   # Follow logs in real-time"
    echo "  PORT=8000 $0 start         # Start on port 8000"
    echo "  $0 stop --force            # Force stop the application"
    echo ""
    echo "Quick Commands:"
    echo "  $0 s                        # Start (shortcut)"
    echo "  $0 t                        # Stop (shortcut)"
    echo "  $0 r                        # Restart (shortcut)"
    echo "  $0 st                       # Status (shortcut)"
    echo "  $0 l                        # Logs (shortcut)"
    echo "  $0 f                        # Follow logs (shortcut)"
}

# Main execution
main() {
    # Check if scripts exist
    check_scripts
    
    # Get command
    local command="${1:-help}"
    
    case "$command" in
        start|s)
            shift
            start_app "$@"
            ;;
        stop|t)
            shift
            stop_app "$@"
            ;;
        restart|r)
            shift
            restart_app "$@"
            ;;
        status|st)
            shift
            show_status "$@"
            ;;
        logs|l)
            shift
            show_logs "${1:-50}"
            ;;
        follow|f)
            shift
            follow_logs "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
