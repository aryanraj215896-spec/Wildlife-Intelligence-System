from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def main() -> None:
    try:
        from src.wildlife_ai.streamlit_app import main as run_app
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "torchvision"}:
            raise
        import streamlit as st

        st.set_page_config(page_title="Wildlife Intelligence setup")
        st.error(f"This Python environment is missing {exc.name}.")
        st.markdown(
            "Run the application with the project virtual environment's Python "
            "interpreter, rather than a global `streamlit` command:"
        )
        st.code(
            r".\.venv\Scripts\python.exe -m pip install -r requirements.txt"
            "\n"
            r".\.venv\Scripts\python.exe -m streamlit run app.py",
            language="powershell",
        )
        st.stop()
    run_app()


if __name__ == "__main__":
    main()
