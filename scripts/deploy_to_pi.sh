#!/usr/bin/env bash
set -euo pipefail

# Raspberry Pi connection settings.
PI_HOSTS=("jozefpro@raspberrypi-5" "raspberrypi-5")
REMOTE_DIR="/home/jozefpro/rpi-street-counter"
BRANCH="main"

RUN_APP="false"

usage() {
  cat <<EOF
Usage: ./scripts/deploy_to_pi.sh [--run]

Deploy the latest committed code to the Raspberry Pi over SSH.

Options:
  --run    Start the Flask app after deployment. Stop it with Ctrl+C.
  -h, --help
           Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_APP="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

echo "Checking local repository state..."

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: this script must be run from inside the project git repository."
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: local repository has uncommitted changes."
  echo "Commit or stash your changes before deploying."
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "Error: git remote 'origin' is not configured."
  exit 1
fi

ORIGIN_URL="$(git remote get-url origin)"

echo "Pushing latest committed code to origin/${BRANCH}..."
git push origin "${BRANCH}"

PI_SSH_TARGET=""
for host in "${PI_HOSTS[@]}"; do
  echo "Checking SSH access to ${host}..."
  if ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" "true"; then
    PI_SSH_TARGET="${host}"
    break
  fi
done

if [[ -z "${PI_SSH_TARGET}" ]]; then
  echo "Error: could not connect to the Raspberry Pi over SSH."
  echo "Tried: ${PI_HOSTS[*]}"
  exit 1
fi

echo "Deploying to ${PI_SSH_TARGET}:${REMOTE_DIR}..."

ssh "${PI_SSH_TARGET}" bash -s -- "${REMOTE_DIR}" "${BRANCH}" "${ORIGIN_URL}" "${RUN_APP}" <<'REMOTE_SCRIPT'
set -euo pipefail

REMOTE_DIR="$1"
BRANCH="$2"
ORIGIN_URL="$3"
RUN_APP="$4"

echo "Using remote directory: ${REMOTE_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "Installing git..."
  sudo apt-get update
  sudo apt-get install -y git
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "Installing python3-venv..."
  sudo apt-get update
  sudo apt-get install -y python3-venv
fi

if [[ ! -d "${REMOTE_DIR}" ]]; then
  echo "Remote directory does not exist. Cloning repository..."
  git clone --branch "${BRANCH}" "${ORIGIN_URL}" "${REMOTE_DIR}"
else
  echo "Remote directory exists. Updating repository without deleting files..."
  cd "${REMOTE_DIR}"

  if [[ ! -d ".git" ]]; then
    echo "Error: ${REMOTE_DIR} exists but is not a git repository."
    echo "No files were deleted. Move or fix this directory manually, then rerun deploy."
    exit 1
  fi

  git fetch origin "${BRANCH}"
  git checkout "${BRANCH}"
  git pull --ff-only origin "${BRANCH}"
fi

cd "${REMOTE_DIR}"

if [[ ! -d ".venv" ]]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
fi

echo "Installing/updating Python dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Deployment finished."
echo
echo "To start the app manually on the Raspberry Pi:"
echo "  cd ${REMOTE_DIR}"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo
echo "Then open the dashboard from another LAN device:"
echo "  http://<rpi-ip>:5000"
echo

if [[ "${RUN_APP}" == "true" ]]; then
  echo "Starting app now. Stop it with Ctrl+C."
  source .venv/bin/activate
  python app.py
fi
REMOTE_SCRIPT
