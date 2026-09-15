import sys
from pathlib import Path

# Make sure "app" is importable regardless of the working directory or how
# this script was invoked (fixes "ModuleNotFoundError: No module named 'app'"
# when run.py isn't executed from its own directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

from app.config import load_config

if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run("app.server:app", host=cfg.server.host, port=cfg.server.port, reload=False)
