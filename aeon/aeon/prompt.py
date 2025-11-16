import importlib
from openai import OpenAI
from pathlib import Path
from string import Template
from typing import Callable

from aeon import prompts
from aeon.decorators import tab_completion
from aeon.logging import logger


def template_varnames(template: Template) -> list[str]:
    """Extract variable names from string.Template object. Assumes we only use $variable syntax,
    not ${variable} syntax.
    """
    return [
        m.group("named")
        for m in Template.pattern.finditer(template.template)
        if m.group("named")
    ]


class Prompt:
    """
    prompt = Prompt("extract_jokes")
    prompt.variables  # See what vars need to be provided to render prompt
    prompt.render(color="blue", shape="triangle")  # Get list of messages with variables filled in.
    prompt.kwargs(color="blue", shape="triangle")  # Get all kwargs to pass to openai api call.
    """

    _default_kwargs = {
        "model": "gpt-4.1-nano",
        "temperature": 0.0,
        "logprobs": True,
    }
    _default_kwargs_gpt_5 = {
        "reasoning_effort": "minimal",
        "verbosity": "low",
    }
    _unsupported_kwargs_gpt_5 = {"logprobs", "top_logprobs", "temperature"}

    def __init__(self, name: str, **kwargs):
        """
        Parameters
        ----------
        name: str
            Prompt name, corresponding to a python file in `aeon.prompts`.
        kwargs: dict
            API call kwargs that will override any defaults provided in the prompt file.
        """
        self.name = name
        self.prompt = importlib.import_module(f"aeon.prompts.{name}")
        self.default_kwargs = self._resolve_kwargs(**self.prompt.kwargs | kwargs)
        if "response_format" not in self.default_kwargs:
            logger.warning(
                f"No response_format specified for prompt {name}. We recommend providing one."
            )

        self.provider = infer_provider(self.default_kwargs["model"])

        # Last message is dynamic, preceding messages are static.
        self.static_messages = self.prompt.messages[:-1]
        self.last_role = self.prompt.messages[-1]["role"]
        self.last_template = Template(self.prompt.messages[-1]["content"])

        # The vars the user must pass in to messages().
        self.variables = template_varnames(self.last_template)

    def _resolve_kwargs(self, **user_kwargs) -> dict:
        """Resolve kwargs from cls defaults, the imported prompt, and the kwargs passed into init.
        Init kwargs take priority over imported prompt kwargs which take priority over cls defaults.
        gpt-5 models have some quirks that we handle here as well, eventually might need to refactor
        if more models turn out to have different/unsupported kwargs.
        """
        # Notice this includes 5.1 variants.
        if "gpt-5" in user_kwargs.get("model", ""):
            defaults = self._default_kwargs_gpt_5
            unsupported = {
                k: v for k, v in user_kwargs.items()
                if k in self._unsupported_kwargs_gpt_5 and v is not None
            }
            if unsupported:
                raise ValueError(f"gpt-5 should not specify these params: {unsupported}")
        else:
            defaults = self._default_kwargs
        return defaults | user_kwargs

    def render(self, **kwargs) -> list[dict]:
        """Rendered `messages` for api call. User must pass in kwargs for all variables in
        `self.variables`. These will be inserted into the last message.
        """
        last_message = {
            "role": self.last_role,
            "content": self.last_template.substitute(**kwargs)
        }
        return self.static_messages + [last_message]

    def kwargs(self, **kwargs) -> dict:
        """Get all kwargs for api call, including rendered `messages`. User must provide kwargs for
        all variables in `self.variables` to insert into the last message.
        """
        return {**self.default_kwargs, "messages": self.render(**kwargs)}

    def __str__(self):
        return f"{type(self).__name__}(name={self.name})"


def infer_provider(model: str) -> str:
    """
    Infer LLM provider name based on model. For now we keep it simple and support just openai and
    openrouter (technically can call openai through openrouter but I believe it's more expensive).
    """
    if "gpt" in model:
        provider = "openai"
    else:
        provider = "openrouter"
    # TODO: add more providers?
    return provider


def list_prompts() -> list[str]:
    """Return aeon's available prompt names."""
    prompt_dir = Path(__file__).parent/"prompts"
    return [path.stem for path in prompt_dir.iterdir() if path.suffix == ".py"]


@tab_completion(list_prompts)
class Prompts:
    """
    Examples
    --------
    # You are creating a prompt. At this point if you hit <tab>, you will see available options.
    Prompt(Prompts.
    """