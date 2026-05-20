from pathlib import Path
import py_compile


def test_streamlit_app_compiles() -> None:
    app_file = Path("semi-marathon-finistere/app.py")
    py_compile.compile(str(app_file), doraise=True)
