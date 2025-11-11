"""
from aeon import prompts

labeler = Labeler(prompt="extract_jokes")
df_labeled = labeler.label(df.id, df.text, output_dir="data/labeled", threads=10)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from tqdm.auto import tqdm
from typing import Union
import traceback
import yaml

from openai import OpenAI
from openai._exceptions import RateLimitError, InternalServerError
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
import pandas as pd

from aeon.config import PROJECT_ROOT
from aeon.logger import logger
from aeon.prompt import Prompt
from aeon.secrets import SecretManager
from aeon.utils import timestamp, git_hash, timer


class LLMLabeler:

    # TODO: consider whether these should be passed in to label method instead? Or if we even need
    # class? IIRC i found myself mostly wanting this to access metadata but maybe if we return it
    # that's sufficient?
    def __init__(
        self,
        prompt: str,
        parent_dir: Union[str, Path] = PROJECT_ROOT/"data/labels",
        **kwargs,
    ):
        SecretManager().set_secrets()
        self.prompt = Prompt(prompt, **kwargs)
        self.client = OpenAI()
        self.parent_dir = Path(parent_dir)
        # Will set this in `label` method.
        self.output_dir = None

    def label(self, df: pd.DataFrame, max_workers: int = 15) -> dict:
        self.output_dir = self.parent_dir/f"{self.prompt.name}/{timestamp()}-{git_hash()}"
        self.output_dir.mkdir(parents=True, exist_ok=False)

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

        df_labeled = pd.DataFrame(results)
        df_labeled = df_labeled.sort_values("i", ascending=True).reset_index(drop=True)
        # TODO: if I often end up returning something like list[Response], this will still be a
        # str instead of dict - we'd need to call json.loads a second time on whatever attr contains
        # the list. Could standardize this and always return a list, even when unnecessary api could
        # always return a list of len 1. Or just leave as is? Or infer. idk
        df_labeled["response_parsed"] = df_labeled["response"].apply(
            lambda x: {} if x is None else json.loads(x["choices"][0]["message"]["content"])
        )
        results["n_errors"] = df_labeled.shape[0] - df_labeled.success.sum()
        results["df"] = df_labeled
        return results

    def _label_one_row(self, i: int, **kwargs) -> dict:
        """
        Parameters
        ----------
        i : int
            Row index. Used for error messages and to ensure we can recover original row order in
            results.
        """
        res = {
            "index": i,
            "success": True,
            "error": "",
            "response": None,
            "api_kwargs": {},
        }
        res["api_kwargs"] = self.prompt.kwargs(**kwargs)
        try:
            response = retryable_api_call(**res["api_kwargs"])
            res["response"] = response.model_dump_json()
        except Exception as e:
            logger.error("row {i} failed with error: {e}")
            res["success"] = False
            res["error"] = traceback.format_exc()

        # Nice to have this if the job fails late or to let us peek at results early.
        with open(self.output_dir/f"{i}.json", "w") as f:
            json.dump(res, f)
        return res


@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((RateLimitError, InternalServerError)),
    wait=wait_chain(wait_fixed(6), wait_fixed(60)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def retryable_api_call(**kwargs):
    """Wrapper to make api call retryable. To keep things simple, we just retry once after
    6 seconds and again after 60 seconds.
    """
    # TODO: maybe add gemini/claude support eventually?
    # Need to confirm if 1) they support "developer" role and 2) pydantic schemas (recall google
    # used typeddict last I checked, but maybe they handled that if supporting openai lib).
    # Useful link on adding google support:
    # https://ai.google.dev/gemini-api/docs/openai
    # And for anthropic:
    # https://docs.claude.com/en/api/openai-sdk
    return self.client.chat.completions.parse(**kwargs)