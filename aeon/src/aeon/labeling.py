"""
from aeon import prompts

labeler = Labeler(prompt="extract_jokes")
df_labeled = labeler.label(df.id, df.text, output_dir="data/labeled", threads=10)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from pathlib import Path
import json
from pydantic import BaseModel
import shutil
from tenacity import (
    retry, stop_after_attempt, wait_exponential, before_sleep_log, retry_if_exception_type,
    wait_chain, wait_fixed
)
from tqdm.auto import tqdm
from typing import Any, Union
import traceback
import yaml

from openai import OpenAI
from openai._exceptions import RateLimitError, InternalServerError
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
import pandas as pd

from aeon.config import DATA_DIR
from aeon.logging import logger
from aeon.prompt import Prompt
from aeon.secrets import SecretManager
from aeon.utils import timestamp, git_hash, timer


SecretManager().set_secrets()

PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1/",
    "openrouter": "https://openrouter.ai/api/v1",
}


def get_client(provider: str) -> OpenAI:
    """Given a name like "openrouter", create the appropriate openai client.
    """
    return OpenAI(
        base_url=PROVIDER_URLS[provider],
        api_key=os.environ[f"{provider.upper()}_API_KEY"]
    )


class LLMLabeler:

    # TODO: consider whether these should be passed in to label method instead? Or if we even need
    # class? IIRC i found myself mostly wanting this to access metadata but maybe if we return it
    # that's sufficient?
    def __init__(
        self,
        prompt_name: str,
        parent_dir: Union[str, Path] = DATA_DIR/"labels",
    ):
        """
        Parameters
        ----------
        prompt_name : str
            Name of aeon prompt to load. aeon.prompt.Prompts provides tab completion for available
            values. (We specify this here on the premise that this will remain relatively fixed
            from run to run, whereas the args we pass to `label` are more likely to vary run to
            run.)
        """
        self.parent_dir = Path(parent_dir)
        self.prompt_name = prompt_name

        # Will set these in `label` method.
        self.client = None
        self.output_dir = None
        self.batch_subdir = None
        self.prompt = None

    def label(
        self,
        df: pd.DataFrame,
        max_workers: int = 15,
        cleanup: bool = True,
        **kwargs
    ) -> dict:
        """
        Parameters
        ----------
        cleanup : bool
            If True, delete `batches` subdir when labeling completes IF a job completes without
            keyboard interrupt. This can reclaim quite a bit of space for large runs and we have
            this data in a parquet anyway.
        kwargs : any
            Forwarded to Prompt (e.g. temperature, model, logprobs).
        """
        self.prompt = Prompt(self.prompt_name, **kwargs)
        self.output_dir = self.parent_dir/f"{self.prompt_name}/{timestamp()}-{git_hash()}"
        self.batch_dir = self.output_dir/"batches"
        self.client = get_client(self.prompt.provider)

        logger.info(f"Labels will be saved in {self.output_dir}")
        self.batch_dir.mkdir(parents=True, exist_ok=False)
        schema_cls = self.prompt.default_kwargs["response_format"]
        with open(self.output_dir/"response_format.json", "w") as f:
            json.dump(schema_cls.model_json_schema(), f)

        missing_vars = set(self.prompt.variables) - set(df.columns)
        if missing_vars:
            raise ValueError(f"df is missing variable(s): {missing_vars}")

        futures = []
        responses = []
        # Pre-fill with dummy values mostly to illustrate to user what values to expect.
        results = {
            "completed": True,
            "duration_seconds": 0.0,
            "df": None,
            "n_errors": 0,
            "output_path": str(self.output_dir/"output.pq"),
        }
        rows = df[self.prompt.variables].to_dict(orient='records')
        progress_bar = tqdm(total=len(rows), desc="Labeling rows")
        with timer() as time:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                try:
                    for i, row in enumerate(rows):
                        futures.append(executor.submit(self._label_one_row, i, **row))
                    for future in as_completed(futures):
                        responses.append(future.result())
                        progress_bar.update(1)  
                except KeyboardInterrupt:
                    logger.info(
                        "Canceling labeling job. Previously launched API calls will still run."
                    )
                    results["completed"] = False
                    for future in futures:
                        future.cancel()
        results["duration_seconds"] = time["duration"]

        # Construct df of results.
        df_labeled = pd.DataFrame(responses)
        df_labeled = df_labeled.sort_values("id", ascending=True).reset_index(drop=True)
        # This is the dynamic message so it's the one we most often want to examine in results.
        df_labeled["last_message"] = df_labeled.api_kwargs.apply(
            lambda x: x['messages'][-1]['content']
        )
        results["n_errors"] = df_labeled.shape[0] - df_labeled.success.sum()
        results["df"] = df_labeled
        df_labeled.to_parquet(results["output_path"])

        logger.info("Removing intermediate results dir since job completed without interruption.")
        if results["completed"] and cleanup:
            shutil.rmtree(self.batch_dir)
        return results

    # TODO: prob need to define openrouter version (they may not support parse() call; also read
    # it's critical to filter out some bad providers, need to check which again); then make cls
    # select the proper api call func from provider.
    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((RateLimitError, InternalServerError)),
        wait=wait_chain(wait_fixed(6), wait_fixed(60)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def retryable_api_call(self, **kwargs) -> tuple[dict, dict]:
        """Wrapper to make api call retryable. To keep things simple, we just retry once after
        6 seconds and again after 60 seconds.

        Parameters
        ----------
        kwargs : any
            Forwarded to openai api call.

        Returns
        -------
        tuple[dict, dict]
            First item is raw response dumped to dict. Second item is just the generated response
            (as a dict because we always specify a pydantic model to generate structured outputs
            and this gets dumped to a dict as well).
        """
        # TODO: maybe add gemini/claude support eventually?
        # Need to confirm if 1) they support "developer" role and 2) pydantic schemas (recall google
        # used typeddict last I checked, but maybe they handled that if supporting openai lib).
        # Useful link on adding google support:
        # https://ai.google.dev/gemini-api/docs/openai
        # And for anthropic:
        # https://docs.claude.com/en/api/openai-sdk
        result = self.client.chat.completions.parse(**kwargs)
        result_dict = result.model_dump(mode="json")
        # TODO: if I often end up returning something like list[Response], second item will still
        # nest response under an "items" key (for example). Could standardize this and ALWAYS
        # return a list, even when unnecessary (api could
        # always return a list of len 1). Or just leave as is? Or infer. idk
        return result_dict, result_dict["choices"][0]["message"]["parsed"]

    def _label_one_row(self, i: int, **kwargs) -> dict:
        """
        Parameters
        ----------
        i : int
            Row index. Used for error messages and to ensure we can recover original row order in
            results.
        """
        res = {
            "id": i,
            "success": True,
            "error": "",
            "response_raw": {},
            "response_content": {},
            "api_kwargs": self.prompt.kwargs(**kwargs),
        }
        try:
            res["response_raw"], res["response_content"] = self.retryable_api_call(
                **res["api_kwargs"]
            )
        except Exception as e:
            logger.error(f"[row {i}] API call failed with error: {e}")
            res["success"] = False
            res["error"] = traceback.format_exc()

        # This is annoying to save in parquet later and we don't need to re-save it for every row.
        res["api_kwargs"].pop("response_format", None)
        try:
            # Nice to have this if the job fails late or to let us peek at results early.
            with open(self.batch_dir/f"{i}.json", "w") as f:
                json.dump(res, f, default=json_dump_default)
        except Exception as e:
            logger.error(f"[row {i}] Save failed with error: {e}")
            res["success"] = False
            res["error"] = traceback.format_exc()
        return res


def json_dump_default(obj: Any):
    """Pass to json.dump to handle objects that otherwise cannot be serialized.
    """
    if issubclass(obj, BaseModel):
        return obj.model_json_schema()
    logger.warning(f"Serializing obj of unexpected type {type(obj)!r}.")
    return str(obj)