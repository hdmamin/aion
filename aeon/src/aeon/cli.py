from importlib import metadata
from pathlib import Path
import shutil

import typer


cli = typer.Typer()


@cli.command()
def version():
    print(metadata.version("aeon"))


@cli.command()
def make_prompt(name: str):
    """
    Parameters
    ----------
    name : str
        Name of prompt. Should be lowercase and snake case, we will do some minimal normalization
        but nothing too rigorous - spaces/dashes/periods will be replaced with underscores.
        Should not include a file suffix.
    """
    file_name = name.lower().replace(" ", "_").replace("-", "_").replace(".", "")
    prompt_dir = Path(__file__)/"prompts"
    file_path = prompt_dir/f"{file_name}.py"
    shutil.copy(prompt_dir/"_template.py", file_path)
    print(f"New prompt template at {file_path} is ready to be updated.")


if __name__ == "__main__":
    cli()