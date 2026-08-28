# AGENTS.md

This file defines repository-level conventions for coding agents.

## Runtime and tooling

- Python version: `3.13` everywhere (local, devcontainer, CI).
- Package manager: `uv` with lockfile (`uv.lock`).
- Create/update the environment with `uv sync`.

## Core commands

- Run app: `uv run streamlit run race_planners/app.py`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Type-check: `uv run mypy tests`
- Test: `uv run pytest`

## Streamlit deployment dependencies

- `race_planners/requirements.txt` is generated, not hand-edited.
- Export command: `scripts/export_streamlit_requirements.sh`
- Validation command: `scripts/check_streamlit_requirements.sh`

## Quality gates

- Install hooks: `uv run pre-commit install`
- Run all hooks: `uv run pre-commit run --all-files`
- CI must run lint, type checks, tests, and requirements export consistency.
