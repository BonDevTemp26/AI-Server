#!/usr/bin/env bash
# =============================================================================
#  setup_remote.sh  —  Deploy AI-Server to a remote machine via SSH + rsync
#
#  What this script does:
#    1. Prompts for SSH connection details (host, user, port, key)
#    2. Verifies SSH connectivity
#    3. Transfers project files with rsync (fast, incremental)
#    4. Remotely installs Docker + Docker Compose (if not already present)
#    5. Pulls Docker images and starts the full stack
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}━━━━  $*  ━━━━${NC}"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
cat <<'BANNER'
   ___  ____      ____                           ____       _
  / _ \|  _|    / ___|  ___ _ ____   _____ _ __|  _ \  ___| |__  _   _
 | | | | |_____\___ \ / _ \ '__\ \ / / _ \ '__| | | |/ _ \ '_ \| | | |
 | |_| | |_____ ___) |  __/ |   \ V /  __/ |  | |_| |  __/ |_) | |_| |
  \___/|_____|  |____/ \___|_|    \_/ \___|_|  |____/ \___|_.__/ \__, |
                                                                   |___/
BANNER
echo -e "${NC}"
echo -e "  ${BOLD}AI-Server — SSH Remote Deployment Script${NC}"
echo -e "  Transfers files and installs Docker on your remote server\n"

# =============================================================================
# STEP 1 — Collect SSH connection details
# =============================================================================
step "Step 1 — SSH Connection Details"

read -rp "  Remote host / IP address         : " REMOTE_HOST
[[ -z "$REMOTE_HOST" ]] && error "Host cannot be empty."

read -rp "  Remote username                  : " REMOTE_USER
[[ -z "$REMOTE_USER" ]] && error "Username cannot be empty."

read -rp "  SSH port                   [22]  : " REMOTE_PORT
REMOTE_PORT="${REMOTE_PORT:-22}"

read -rp "  SSH key path  [~/.ssh/id_rsa]    : " SSH_KEY
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
[[ ! -f "$SSH_KEY" ]] && error "SSH key not found at: $SSH_KEY"

read -rp "  Remote deploy path [~/AI-Server] : " REMOTE_DIR
REMOTE_DIR="${REMOTE_DIR:-~/AI-Server}"

# Convenience SSH/rsync options
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${REMOTE_PORT} -i ${SSH_KEY}"
RSYNC_SSH="ssh ${SSH_OPTS}"

info "Target: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}  ->  ${REMOTE_DIR}"

# =============================================================================
# STEP 2 — Verify SSH connectivity
# =============================================================================
step "Step 2 — Verifying SSH Connectivity"

if ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "echo 'SSH_OK'" 2>/dev/null | grep -q "SSH_OK"; then
    success "SSH connection successful."
else
    error "Cannot connect to ${REMOTE_USER}@${REMOTE_HOST}. Check your credentials and try again."
fi

# =============================================================================
# STEP 3 — Transfer project files via rsync
# =============================================================================
step "Step 3 — Transferring Project Files (rsync)"

# Local project root (directory containing this script)
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info "Source : ${LOCAL_DIR}"
info "Dest   : ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
echo ""

# Create the remote directory first
ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

# rsync — exclude large AI model weights, git metadata, Python caches,
# Docker build artefacts, and secrets (.env).
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='backend/surveillance.db' \
    --exclude='backend/uploads/' \
    --exclude='*.pt' \
    --exclude='*.onnx' \
    -e "${RSYNC_SSH}" \
    "${LOCAL_DIR}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

success "File transfer complete."

# ── Transfer .env separately (sensitive) ─────────────────────────────────────
step "Step 3b — Transferring .env (environment variables)"
warn ".env contains secrets. It will be transferred with restricted permissions (600)."
read -rp "  Transfer .env now? [Y/n]: " TRANSFER_ENV
TRANSFER_ENV="${TRANSFER_ENV:-Y}"
if [[ "${TRANSFER_ENV^^}" == "Y" ]]; then
    rsync -avz --chmod=F600 \
        -e "${RSYNC_SSH}" \
        "${LOCAL_DIR}/.env" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/.env"
    success ".env transferred with permissions 600."
else
    warn "Skipped .env — copy it manually before starting the stack."
fi

# ── Transfer YOLO / RT-DETR model weights separately ─────────────────────────
step "Step 3c — Model Weights (.pt files)"
info "Large model files (*.pt) were excluded from the main sync to save time."
read -rp "  Transfer model weights now? (may take a while) [Y/n]: " TRANSFER_MODELS
TRANSFER_MODELS="${TRANSFER_MODELS:-Y}"
if [[ "${TRANSFER_MODELS^^}" == "Y" ]]; then
    info "Transferring model weights..."
    rsync -avz --progress \
        -e "${RSYNC_SSH}" \
        "${LOCAL_DIR}"/backend/*.pt \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/backend/" 2>/dev/null \
        || warn "No .pt files found locally or transfer failed."
    success "Model weights transferred."
else
    warn "Skipped model weights — place them in backend/ on the remote before starting."
fi

# =============================================================================
# STEP 4 — Install Docker + Docker Compose on remote
# =============================================================================
step "Step 4 — Installing Docker & Docker Compose on Remote"

ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" bash <<'REMOTE_INSTALL'
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[REMOTE]${NC} $*"; }
success() { echo -e "${GREEN}[REMOTE-OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[REMOTE-WARN]${NC} $*"; }

# ── Docker ──────────────────────────────────────────────────────────────────
if command -v docker &>/dev/null; then
    success "Docker already installed: $(docker --version)"
else
    info "Installing Docker..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl gnupg lsb-release

    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin

    sudo systemctl enable docker --now
    success "Docker installed successfully."
fi

# ── Add user to docker group ─────────────────────────────────────────────────
if ! groups "$USER" | grep -q '\bdocker\b'; then
    warn "Adding ${USER} to the docker group (re-login required to take effect)."
    sudo usermod -aG docker "$USER"
fi

# ── Docker Compose ───────────────────────────────────────────────────────────
if docker compose version &>/dev/null; then
    success "Docker Compose available: $(docker compose version)"
elif command -v docker-compose &>/dev/null; then
    success "docker-compose (standalone) available: $(docker-compose --version)"
else
    info "Installing Docker Compose plugin..."
    sudo apt-get install -y -qq docker-compose-plugin
    success "Docker Compose installed."
fi

# ── rsync ────────────────────────────────────────────────────────────────────
if ! command -v rsync &>/dev/null; then
    info "Installing rsync..."
    sudo apt-get install -y -qq rsync
fi

success "All remote dependencies are ready."
REMOTE_INSTALL

success "Remote dependency installation complete."

# =============================================================================
# STEP 5 — Start the Docker stack on remote
# =============================================================================
step "Step 5 — Starting AI-Server Stack on Remote"

read -rp "  Start the Docker Compose stack now? [Y/n]: " START_STACK
START_STACK="${START_STACK:-Y}"

if [[ "${START_STACK^^}" == "Y" ]]; then
    ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" bash <<REMOTE_START
        set -euo pipefail
        cd ${REMOTE_DIR}

        echo "  Building backend image..."
        sudo docker compose build backend

        echo "  Pulling other service images..."
        sudo docker compose pull --ignore-buildable 2>/dev/null || true

        echo "  Starting all services..."
        sudo docker compose up -d

        echo ""
        echo "  ---- Service Status ----"
        sudo docker compose ps
REMOTE_START
    success "Stack started on ${REMOTE_HOST}."
else
    warn "Stack not started. To start it manually:"
    echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} -p ${REMOTE_PORT} -i ${SSH_KEY}"
    echo "  cd ${REMOTE_DIR} && sudo docker compose up -d"
fi

# =============================================================================
# DONE
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}================================================${NC}"
echo -e "${BOLD}${GREEN}  Deployment Complete!${NC}"
echo -e "${BOLD}${GREEN}================================================${NC}"
echo ""
echo -e "  ${BOLD}Services on ${REMOTE_HOST}:${NC}"
echo -e "  Frontend  ->  http://${REMOTE_HOST}:80"
echo -e "  Backend   ->  http://${REMOTE_HOST}:8000"
echo -e "  RTSP      ->  rtsp://${REMOTE_HOST}:8554"
echo -e "  WebRTC    ->  http://${REMOTE_HOST}:8889"
echo -e "  Postgres  ->  ${REMOTE_HOST}:5432"
echo ""
echo -e "  ${YELLOW}Tip:${NC} If Docker commands fail with permission errors on the remote,"
echo -e "  log out and back in to activate the 'docker' group membership."
echo ""
