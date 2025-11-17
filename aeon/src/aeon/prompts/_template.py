from pydantic import BaseModel, Field


class Response(BaseModel):
    # TODO: update expected fields an descriptions.
    foo: str = Field(
        ...,
        description=""
    )


# TODO: this is only necessary if you want list[dict] of results. If not, just define Response
# and update kwargs response_format further down.
class BatchResponse(BaseModel):
    items: list[Response]


# TODO: fill in developer message content
messages = [
    {
        "role": "developer",
        "content": """
<instructions>
<example_input>
</example_input>
<example_output>
[
    {
    },
]
</example_output>
<explanation>
</explanation>
</instructions>
"""
    },
    {
        "role": "user",
        "content": "$input_text"  # TODO: each $var here maps to one col in the input df you provide
    }
]

kwargs = {
    # TODO: remember gpt-5 models don't have logprobs, if you want those then use something else
    "model": "gpt-5-mini",
    "response_format": BatchResponse,
}