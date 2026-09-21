#!/usr/bin/env bash
#=============================================================================
#region SCRIPT DETAILS
#=============================================================================
#.SYNOPSIS
#Checks for macOS OS, verifies prerequisites,
#and installs required Python packages for Assignment 2.
#
#.DESCRIPTION
#- Verifies Python 3.8+ is installed
#- Upgrades pip to the latest version
#- Detects imports dynamically from the assignment script
#- Installs mapped packages
#- Offers to launch the assignment script after setup
#
#.NOTES
#Run from the folder containing the .py file.
#=============================================================================
#endregion
#=============================================================================
#region Prerequisites
#=============================================================================
#  OS CHECK: Only run on macOS (Darwin)
OS_NAME=$(uname)
if [ "$OS_NAME" != "Darwin" ]; then
    echo "OS is not macOS. This script is only intended for macOS devices."
    exit 666
fi
#=============================================================================
#endregion
#=============================================================================
#region FUNCTIONS
#=============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m';  GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m';     RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}$*${RESET}\n"; }

# ── Change to the script's own directory ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ASSIGNMENT_VIEWER="scripts/Assignment_02_Attention_Under_The_Hood.py"

clear
echo -e "${BOLD}=================================================================${RESET}"
echo -e "${BOLD}   Assignment 2 Attention Under the Hood — macOS Installer       ${RESET}"
echo -e "${BOLD}=================================================================${RESET}"
echo ""
#=============================================================================
#endregion
#=============================================================================
#region EXECUTION
#=============================================================================
#Step 1: Find Python 3.8+
#=============================================================================
header "Step 1: Locating Python 3.8+"

PYTHON=""
MIN_MAJOR=3
MIN_MINOR=8

try_python() {
    local cmd="$1"
    if command -v "$cmd" &>/dev/null; then
        local ver
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || return 1
        local major minor
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge "$MIN_MAJOR" ] && [ "$minor" -ge "$MIN_MINOR" ]; then
            PYTHON="$cmd"
            return 0
        fi
    fi
    return 1
}

for cmd in python3 python3.13 python3.12 python3.11 python3.10 python3.9 python3.8 python; do
    if try_python "$cmd"; then
        PY_VER=$("$PYTHON" -c "import sys; print(sys.version.split()[0])")
        success "Found: $PYTHON  (Python $PY_VER)"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python ${MIN_MAJOR}.${MIN_MINOR}+ was not found on this system."
    echo ""
    echo -e "  ${BOLD}Install Python from one of these sources:${RESET}"
    echo "    • Official:  https://www.python.org/downloads/"
    echo "    • Homebrew:  brew install python"
    echo ""
    echo "  After installing Python, re-run this script."
    echo ""
    read -rp "Press Enter to exit..." _
    exit 1
fi

#=============================================================================
#Step 2: Upgrade pip
#=============================================================================
header "Step 2: Ensuring pip is up to date"

"$PYTHON" -m ensurepip --upgrade --quiet 2>/dev/null || true
"$PYTHON" -m pip install --upgrade pip --quiet
success "pip is ready"

#=============================================================================
#Step 3: Check application files
#=============================================================================
header "Step 3: Checking application files"

if [ -f "$ASSIGNMENT_VIEWER" ]; then
    success "Found: $ASSIGNMENT_VIEWER"
else
    error "Not found: $ASSIGNMENT_VIEWER"
    warn "This file must be in the same folder as the installer to detect imports."
    echo ""
    read -rp "Press Enter to exit..." _
    exit 1
fi

#=============================================================================
#Step 4: Detect & install required packages
#=============================================================================
header "Step 4: Detecting & installing packages"
info "Parsing $ASSIGNMENT_VIEWER for imports..."

IMPORTS=()
while IFS= read -r import_name; do
  IMPORTS+=("$import_name")
done < <("$PYTHON" -c "
import ast
from pathlib import Path

source = Path(r'''$SCRIPT_DIR/$ASSIGNMENT_VIEWER''').read_text(encoding='utf-8')
magic_prefixes = (chr(33), chr(37))
source = '\n'.join('' if line.lstrip()[:1] in magic_prefixes else line for line in source.splitlines())
tree = ast.parse(source)
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            imports.add(alias.name.split('.')[0])
    elif isinstance(node, ast.ImportFrom) and node.module:
        imports.add(node.module.split('.')[0])
for name in sorted(imports):
    print(name)
")

declare -A PACKAGE_MAP=(
  [torch]="torch"
  [numpy]="numpy"
  [matplotlib]="matplotlib"
  [seaborn]="seaborn"
)

STANDARD_LIBRARY=(math pathlib)

PACKAGES=()
for import_name in "${IMPORTS[@]}"; do
  if [[ " ${STANDARD_LIBRARY[*]} " =~ " ${import_name} " ]]; then
    continue
  fi
  if [ -n "${PACKAGE_MAP[$import_name]+x}" ]; then
    pkg="${PACKAGE_MAP[$import_name]}"
    if [[ ! " ${PACKAGES[*]} " =~ " ${pkg} " ]]; then
      PACKAGES+=("$pkg")
    fi
  else
    error "Unmapped third-party import detected: $import_name"
    info "Add it to the package map before continuing."
    exit 1
  fi
done

success "Detected packages needed: ${PACKAGES[*]}"

if [ "${#PACKAGES[@]}" -gt 0 ]; then
  echo ""
  info "Installing dependencies..."
  "$PYTHON" -m pip install --quiet "${PACKAGES[@]}"
  success "Installation complete."
else
  success "No mapped third-party packages detected."
fi

#=============================================================================
#Step 5: Verify imports
#=============================================================================
header "Step 5: Verifying imports"

VERIFY_FAILED=0
for import_name in torch numpy matplotlib seaborn; do
  if "$PYTHON" -c "import $import_name" &>/dev/null; then
    success "$import_name successfully imported"
  else
    error "Failed to import: $import_name"
    VERIFY_FAILED=1
  fi
done

if [ "$VERIFY_FAILED" -eq 1 ]; then
    echo ""
    error "One or more packages failed verification."
    read -rp "Press Enter to exit..." _
    exit 1
fi

#=============================================================================
#Step 6: Launch prompt
#=============================================================================
echo ""
echo -e "${BOLD}=================================================================${RESET}"
echo -e "${BOLD}   Setup Complete — Assignment 2 Attention Under the Hood     ✓${RESET}"
echo -e "${BOLD}=================================================================${RESET}"
echo ""
echo -e "  ${BOLD}1${RESET}  Launch $ASSIGNMENT_VIEWER"
echo -e "  ${BOLD}2${RESET}  Exit without launching"
echo ""

while true; do
    read -rp "  Enter your choice [1/2]: " choice
    case "$choice" in
        1)
            info "Launching $ASSIGNMENT_VIEWER ..."
            "$PYTHON" "$ASSIGNMENT_VIEWER"
            break
            ;;
        2)
            info "Exiting. Run this script again any time to launch."
            break
            ;;
        *)
            warn "Please enter 1 or 2."
            ;;
    esac
done

echo ""
#=============================================================================
#endregion
#=============================================================================
