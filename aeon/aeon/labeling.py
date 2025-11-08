"""
from aeon import prompts

labeler = Labeler(prompt="extract_jokes")
df_labeled = labeler.label(df.id, df.text, output_dir="data/labeled", threads=10)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import traceback
import yaml

from openai import OpenAI
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
import pandas as pd

from aeon.logger import logger
from aeon.prompt import Prompt
from aeon.secrets import SecretManager


class LLMLabeler:

    # TODO: consider whether these should be passed in to label method instead? Or if we even need
    # class? IIRC i found myself mostly wanting this to access metadata but maybe if we return it
    # that's sufficient?
    def __init__(self, prompt: str, **kwargs):
        SecretManager().set_secrets()
        self.prompt = Prompt(prompt, **kwargs)
        self.client = OpenAI()

    def label(self, df: pd.DataFrame, max_workers: int = 15) -> dict:
        missing_vars = set(self.prompt.variables) - set(df.columns)
        if missing_vars:
            raise ValueError(f"df is missing variable(s): {missing_vars}")

        futures = []
        responses = []
        results = {
            "completed": True
        }
        rows = df[self.prompt.variables].to_dict(orient='records')
        progress_bar = tqdm(total=len(rows), desc="Labeling rows")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                for i, row in enumerate(rows):
                    futures.append(executor.submit(self._label_one_row, i, **row))
                for future in as_completed(futures):
                    responses.append(future.result())
                    progress_bar.update(1)  
            except KeyboardInterrupt:
                logger.info("Canceling labeling job. Previously launched API calls will still run.")
                results["completed"] = False
                for future in futures:
                    future.cancel()

        df_labeled = pd.DataFrame(results)
        df_labeled = df_labeled.sort_values("i", ascending=True).reset_index(drop=True)
        # TODO: if I often end up returning something like list[Response], this will still be a
        # str instead of dict - we'd need to call json.loads a second time on whatever attr contains
        # the list. Could standardize this and always return a list, even when unnecessary api could
        # always return a list of len 1. Or just leave as is? Or infer. idk
        df_labeled["response_parsed"] = df_labeled["response"].apply(
            lambda x: {} if x is None else json.loads(x.choices[0].message.content)
        )
        results["df"] = df_labeled
        return results

    # TODO: maybe save this somewhere as well in case job fails late. Still determining whether it's
    # fine to do here or if that will slow things down much. Could also save in `label`, e.g.
    # every n'th iteration in as_completed (perhaps keeping separate queues of saved and unsaved)
    # or having another thread running in the bg saving periodically (maybe queue is also useful
    # here)
    def _label_one_row(self, i: int, **kwargs) -> dict:
        res = {
            "index": i,
            "success": True,
            "error": "",
            "response": None,
        }
        try:
            api_kwargs = self.prompt.kwargs(**kwargs)
            res["response"] = self.client.chat.completions.parse(
                **api_kwargs
            )
        except Exception as e:
            logger.error("row {i} failed with error: {e}")
            res["success"] = False
            res["error"] = traceback.format_exc()
        return res