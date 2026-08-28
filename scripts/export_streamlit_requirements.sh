#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_FILE="race_planners/requirements.txt"

cd "${ROOT_DIR}"

uv export --no-hashes --no-dev --format requirements-txt -o "${OUTPUT_FILE}"
