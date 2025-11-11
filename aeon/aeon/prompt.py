import importlib
from string import Template

from aeon import prompts
from aeon.logging import logger


def template_varnames(template: Template) -> list[str]:
    """Extract variable names from string.Template object. Assumes we only use $variable syntax,
    not ${variable} syntax.
    """
    return [
        match.group("named")
        for match in Template.pattern.finditer(template.template)
        if match.group("named")
    ]


class Prompt:
    """
    prompt = Prompt("extract_jokes")
    prompt.variables  # See what vars need to be provided to render prompt
    prompt.render(color="blue", shape="triangle")  # Get list of messages with variables filled in.
    prompt.kwargs(color="blue", shape="triangle")  # Get all kwargs to pass to openai api call.
    """

    default_kwargs = {
        "model": "gpt-4.1-nano",
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 10,
    }

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
        self.kwargs = self.default_kwargs | self.prompt.kwargs | **kwargs
        if "response_format" not in self.kwargs:
            logger.warning(
                f"No response_format specified for prompt {name}. We recommend providing one."
            )

        # Last message is dynamic, preceding messages are static.
        self.static_messages = self.prompt.messages[:-1]
        self.last_role = self.prompt.messages[-1]["role"]
        self.last_template = Template(self.prompt.messages[-1]["content"])

        # The vars the user must pass in to messages().
        self.variables = template_varnames(self.last_template)

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
        return {**self.kwargs, "messages": self.render(**kwargs)}

    def __str__(self):
        return f"{type(self).__name__}(name={self.name})"