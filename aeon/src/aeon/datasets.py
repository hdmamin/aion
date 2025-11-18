import os

from huggingface_hub import login, HfApi
import pandas as pd

from aeon import config
from aeon.secrets import SecretManager


def save_dataset(df: pd.DataFrame, name: str, upload_to_hub: bool = True) -> None:
    """Save a dataset to local parquet in {project_root}/data/datasets and optionally create a 
    Huggingface Hub dataset in my hmamin/aeon collection.
    """
    out_dir = config.DATRA_DIR/f"datasets/{name}"
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(out_dir/"df.pq")

    if hub:
        if not hub_token:
            raise ValueError("hub_token must be provided when upload_to_hub is True.")

        secrets = SecretManager().get_secrets()
        login(secrets["HUGGINGFACE_TOKEN"])
        hf_api = HfApi()
        hf_api.add_collection_item(
            collection_slug="hmamin/aeon",
            item_id=f"hmamin/{name}",
            item_type="dataset"
        )