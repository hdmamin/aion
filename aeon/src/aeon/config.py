from pathlib import Path


# The dir containing `src` and `data`.
LIB_ROOT = Path(__file__).parent.parent.parent
# The dir containing both aeon and nanochat libs.
PROJECT_ROOT = LIB_ROOT.parent
DATA_DIR = LIB_ROOT/"data"