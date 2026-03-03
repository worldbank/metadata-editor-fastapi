#!/bin/bash
# =============================================================
# install-service.sh
#
# Installs editor-fastapi as a systemd service on Linux.
# Must be run as root (or with sudo).
#
# REQUIRED — set before running:
#   CONDA_PYTHON_PATH   Absolute path to the Python executable in your
#                       conda environment. Find it with:
#                         conda activate metadata-editor
#                         which python
#
# OPTIONAL overrides (defaults shown):
#   PROJECT_DIR         Path to the application root  (default: directory containing this script's parent)
#   SERVICE_USER        OS user the service runs as   (default: editor-fastapi)
#   HOST                Bind address                  (default: 0.0.0.0)
#   PORT                Port                          (default: 8000)
#   SHARED_GROUP        Group shared with web server  (default: editor-shared)
#   WEB_SERVER_USER     Apache/NGINX user to add to shared group (default: www-data)
#   STORAGE_PATH        Shared folder path; permissions set if provided (default: unset)
#
# Usage:
#   sudo CONDA_PYTHON_PATH=/home/myuser/miniconda3/envs/metadata-editor/bin/python \
#        bash deploy/linux/install-service.sh
#
#   sudo bash deploy/linux/install-service.sh --uninstall
# =============================================================

set -e

SERVICE_NAME="editor-fastapi"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
TEMPLATE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/editor-fastapi.service"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SERVICE_USER="${SERVICE_USER:-editor-fastapi}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
SHARED_GROUP="${SHARED_GROUP:-editor-shared}"
WEB_SERVER_USER="${WEB_SERVER_USER:-www-data}"
STORAGE_PATH="${STORAGE_PATH:-}"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# -------------------------------------------------------
check_root() {
    if [ "$EUID" -ne 0 ]; then
        error "This script must be run as root. Use: sudo CONDA_PYTHON_PATH=... bash $0"
    fi
}

# -------------------------------------------------------
validate_inputs() {
    # CONDA_PYTHON_PATH is mandatory — no guessing
    if [ -z "${CONDA_PYTHON_PATH:-}" ]; then
        error "CONDA_PYTHON_PATH is not set.

Find the correct path by running (as the user who owns the conda environment):

    conda activate metadata-editor
    which python

Then re-run this script with that path:

    sudo CONDA_PYTHON_PATH=/path/to/python bash $0"
    fi

    if [ ! -x "$CONDA_PYTHON_PATH" ]; then
        error "CONDA_PYTHON_PATH does not exist or is not executable: $CONDA_PYTHON_PATH"
    fi

    # Verify uvicorn is importable with this Python
    if ! "$CONDA_PYTHON_PATH" -c "import uvicorn" 2>/dev/null; then
        error "uvicorn is not installed in the environment at: $CONDA_PYTHON_PATH
Activate the conda environment and run: pip install -r requirements.txt"
    fi

    if [ ! -f "$PROJECT_DIR/main.py" ]; then
        error "main.py not found in PROJECT_DIR: $PROJECT_DIR
Set PROJECT_DIR to the application root directory."
    fi

    if [ ! -f "$TEMPLATE_FILE" ]; then
        error "Service template not found: $TEMPLATE_FILE"
    fi

    # Derive lib path from the Python executable: <env>/bin/python -> <env>/lib
    CONDA_LIB_PATH="$(dirname "$(dirname "$CONDA_PYTHON_PATH")")/lib"
    if [ ! -d "$CONDA_LIB_PATH" ]; then
        warn "Expected conda lib directory not found: $CONDA_LIB_PATH"
        warn "LD_LIBRARY_PATH will be set to this path anyway — verify it is correct."
    fi

    info "Configuration:"
    info "  CONDA_PYTHON_PATH : $CONDA_PYTHON_PATH"
    info "  CONDA_LIB_PATH    : $CONDA_LIB_PATH"
    info "  PROJECT_DIR       : $PROJECT_DIR"
    info "  SERVICE_USER      : $SERVICE_USER"
    info "  HOST:PORT         : $HOST:$PORT"
    info "  SHARED_GROUP      : $SHARED_GROUP"
    info "  WEB_SERVER_USER   : $WEB_SERVER_USER"
    if [ -n "$STORAGE_PATH" ]; then
    info "  STORAGE_PATH      : $STORAGE_PATH"
    else
    info "  STORAGE_PATH      : (not set — skip permission setup)"
    fi
}

# -------------------------------------------------------
setup_shared_permissions() {
    info "Setting up shared group for web server access..."

    # Create shared group if it doesn't exist
    if getent group "$SHARED_GROUP" > /dev/null 2>&1; then
        info "Shared group '$SHARED_GROUP' already exists"
    else
        groupadd "$SHARED_GROUP"
        success "Created shared group '$SHARED_GROUP'"
    fi

    # Add FastAPI service user to the shared group
    usermod -aG "$SHARED_GROUP" "$SERVICE_USER"
    success "Added '$SERVICE_USER' to group '$SHARED_GROUP'"

    # Add web server user to the shared group (if the user exists)
    if id "$WEB_SERVER_USER" > /dev/null 2>&1; then
        usermod -aG "$SHARED_GROUP" "$WEB_SERVER_USER"
        success "Added '$WEB_SERVER_USER' to group '$SHARED_GROUP'"
    else
        warn "Web server user '$WEB_SERVER_USER' not found — skipping."
        warn "Add it manually once installed: sudo usermod -aG $SHARED_GROUP <web-user>"
    fi

    # Set permissions on the shared storage folder
    if [ -n "$STORAGE_PATH" ]; then
        if [ ! -d "$STORAGE_PATH" ]; then
            error "STORAGE_PATH does not exist: $STORAGE_PATH"
        fi
        # Own the directory by the web server user + shared group
        chown -R "$WEB_SERVER_USER:$SHARED_GROUP" "$STORAGE_PATH"
        # setgid (2) ensures new files inherit the shared group
        # 2775 = rwxrwsr-x: owner+group full access, others read/execute
        chmod -R 2775 "$STORAGE_PATH"
        success "Set permissions on STORAGE_PATH: $STORAGE_PATH"
        info "  Owner: $WEB_SERVER_USER:$SHARED_GROUP  Mode: 2775 (setgid)"
    else
        warn "STORAGE_PATH not set — skipping directory permission setup."
        warn "Set it manually after install:"
        warn "  sudo chown -R $WEB_SERVER_USER:$SHARED_GROUP /path/to/storage"
        warn "  sudo chmod -R 2775 /path/to/storage"
    fi
}

# -------------------------------------------------------
create_service_user() {
    if id "$SERVICE_USER" &>/dev/null; then
        info "Service user '$SERVICE_USER' already exists"
    else
        info "Creating service user '$SERVICE_USER'..."
        useradd -r -s /bin/false -d "$PROJECT_DIR" "$SERVICE_USER"
        success "Created user '$SERVICE_USER'"
    fi

    mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/jobs"
    chown "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR/logs" "$PROJECT_DIR/jobs"
}

# -------------------------------------------------------
install_service() {
    info "Generating systemd unit file from template..."

    sed \
        -e "s|%%CONDA_PYTHON_PATH%%|${CONDA_PYTHON_PATH}|g" \
        -e "s|%%CONDA_LIB_PATH%%|${CONDA_LIB_PATH}|g" \
        -e "s|%%PROJECT_DIR%%|${PROJECT_DIR}|g" \
        -e "s|%%SERVICE_USER%%|${SERVICE_USER}|g" \
        -e "s|%%HOST%%|${HOST}|g" \
        -e "s|%%PORT%%|${PORT}|g" \
        -e "s|%%SHARED_GROUP%%|${SHARED_GROUP}|g" \
        "$TEMPLATE_FILE" > "$SERVICE_FILE"

    chmod 644 "$SERVICE_FILE"
    success "Installed: $SERVICE_FILE"

    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
    success "Service enabled (will start at boot)"

    systemctl start "$SERVICE_NAME"

    sleep 2
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        success "Service is running"
        echo ""
        info "Useful commands:"
        info "  systemctl status  $SERVICE_NAME"
        info "  systemctl restart $SERVICE_NAME"
        info "  systemctl stop    $SERVICE_NAME"
        info "  journalctl -u $SERVICE_NAME -f      # live logs"
    else
        warn "Service failed to start. Recent logs:"
        journalctl -u "$SERVICE_NAME" --no-pager -n 30 || true
        exit 1
    fi
}

# -------------------------------------------------------
uninstall_service() {
    info "Uninstalling $SERVICE_NAME service..."

    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        systemctl stop "$SERVICE_NAME"
        success "Service stopped"
    fi

    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        systemctl disable "$SERVICE_NAME"
        success "Service disabled"
    fi

    if [ -f "$SERVICE_FILE" ]; then
        rm "$SERVICE_FILE"
        systemctl daemon-reload
        success "Removed: $SERVICE_FILE"
    else
        warn "Service file not found: $SERVICE_FILE"
    fi

    success "Uninstall complete"
}

# -------------------------------------------------------
main() {
    check_root

    case "${1:-}" in
        --uninstall)
            uninstall_service
            ;;
        "")
            info "=== Installing $SERVICE_NAME as a systemd service ==="
            validate_inputs
            create_service_user
            setup_shared_permissions
            install_service
            success "=== Installation complete ==="
            ;;
        *)
            echo "Usage: sudo CONDA_PYTHON_PATH=... bash $0 [--uninstall]"
            exit 1
            ;;
    esac
}

main "$@"
