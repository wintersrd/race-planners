# race-planners

Race planning applications and tools, currently focused on a Streamlit app for the Semi-Marathon du Finistere.

## Python and environment

- Python: `3.13`
- Dependency and environment manager: `uv`
- Virtual environment: `.venv` at repository root

Setup:

```bash
uv sync
uv run pre-commit install
```

## Run the app

Run Streamlit from repository root so path behavior matches Streamlit Community Cloud:

```bash
uv run streamlit run semi-marathon-finistere/app.py
```

## Quality commands

```bash
uv run ruff check .
uv run ruff format .
uv run mypy tests
uv run pytest
uv run pre-commit run --all-files
```

## Streamlit deployment dependencies

`semi-marathon-finistere/requirements.txt` is generated from the `uv` project state.

```bash
scripts/export_streamlit_requirements.sh
scripts/check_streamlit_requirements.sh
```

Do not hand-edit `semi-marathon-finistere/requirements.txt`.

## Devcontainer

The repository includes `.devcontainer/devcontainer.json` with Python 3.13 and `uv` workflow parity.
