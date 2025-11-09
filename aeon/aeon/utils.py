from contextlib import contextmanager
from datetime import datetime
import subprocess
import time

from aeon.logging import logger


def timestamp() -> str:
    """Format current timestamp as YYYY.MM.DD_HH.MM.SS."""
    return datetime.now().strftime("%Y.%m.%d_%H.%M.%S")


def uncommitted_changes() -> bool:
    """Check if there are any uncommitted changes in the current git repository.
    This also returns True if there are unstaged changes because those are also uncommitted."

    Note: this assumes we are using the editable install inside my larger aeon repo. If this was
    installed norally in site-packages, this would not work as intended.
    """
    return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())


def git_hash() -> str:
    """
    Get the latest git commit hash from the current repo.

    Note: this assumes we are using the editable install inside my larger aeon repo. If this was
    installed norally in site-packages, this would not work as intended.
    """
    if uncommitted_changes():
        logger.warning(
            "Unstaged/uncommitted changes in git repository. You might want to commit them before "
            "running."
        )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


@contextmanager
def timer(name: str = "BLOCK") -> dict:
    """Time how long a block of code takes to execute.
    """
    try:
        res = {"start": time.perf_counter()}
        yield res
    finally:
        res["duration"] = time.perf_counter() - start
        logger.info(f"[TIMER] {name} executed in {res['duration']:.3f} s.")