from datetime import datetime
import subprocess


def timestamp() -> str:
    """Format current timestamp as YYYY.MM.DD_HH.MM.SS."""
    return datetime.now().strftime("%Y.%m.%d_%H.%M.%S")


def git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()