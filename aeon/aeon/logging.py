import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Necessary to get this to show up in Jupyter.
logger.addHandler(logging.StreamHandler())