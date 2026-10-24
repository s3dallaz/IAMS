"""
Start the application: the session scheduler first, then the Streamlit
server in this same process.

With a plain `streamlit run app.py` the scheduler only starts when the first
page loads. Started from here, scheduled sessions start and close on time
from the moment the server runs, even if nobody ever opens the page.

    python run_app.py [streamlit options, e.g. --server.headless true]
"""

import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "app.py"


def main() -> None:
    from src.services.session_scheduler import start_session_scheduler
    from streamlit.web import cli

    start_session_scheduler()
    sys.argv = ["streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    cli.main(prog_name="streamlit")


if __name__ == "__main__":
    main()
