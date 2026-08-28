#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_FILE="${ROOT_DIR}/race_planners/requirements.txt"
TMP_FILE="$(mktemp)"

cleanup() {
  rm -f "${TMP_FILE}"
}

trap cleanup EXIT

cd "${ROOT_DIR}"

uv export --no-hashes --no-dev --format requirements-txt -o "${TMP_FILE}" >/dev/null

if ! python - "${EXPECTED_FILE}" "${TMP_FILE}" <<'PY'
import pathlib
import sys

expected = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
actual = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()

def normalize(lines: list[str]) -> list[str]:
    return [line for line in lines if not line.startswith("#    uv export ")]

if normalize(expected) != normalize(actual):
    raise SystemExit(1)
PY
then
  diff -u "${EXPECTED_FILE}" "${TMP_FILE}" || true
  printf "\nrequirements.txt is out of sync. Run scripts/export_streamlit_requirements.sh\n" >&2
  exit 1
fi
