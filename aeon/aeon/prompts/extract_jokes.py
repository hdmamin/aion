from pydantic import BaseModel, Field


class Response(BaseModel):
    joke: str = Field(
        ...,
        description="One or more sentences containing a joke from the input text. In most cases "
                    "this should quote the source verbatim, but you may apply minor cleanup to "
                    "remove transcription artifacts, filler words like 'uh', etc. "
                    "The transcripts are from live spoken performances but this field is "
                    "designed to be *read*."
    )

    prompt: str = Field(
        ...,
        description="A one line comment or question from an imaginary interlocutor that could "
                    " elicit this joke in conversation. Another framing: what might someone say to "
                    "a funny chatbot that could plausibly elicit the relevant `joke`?"
    )
        
    subtext: str = Field(
        ...,
        description="The often banal observation underlying the joke. This should not be funny."
    )
        
    unfunny_variant: str = Field(
        ...,
        description="Lightly tweak or rearrange the wording of the original joke such that it "
                    "is not longer funny, while still maintaining the same subtext. Change the "
                    "joke only as much as is necessary to remove the humor, and no further."
    )


class BatchResponse(BaseModel):
    items: list[Response]


messages = [
    {
        "role": "developer",
        "content": """
<instructions>
You are a detail-oriented research assistant with a shrewd understanding of humor, human behavior, and writing. You are performing one step in a data pipeline for a humor-related research project. Your task is to extract all jokes from the input passage and return valid JSON where each object contains "joke" and "subtext" fields. You can think of the subtext as the often banal point that the joke is making, which the comedian has ultimately massaged or restructured such that the resulting joke subverts the audience's expectations in a fun way. The subtext should not be funny, e.g. it might plainly state "airplane food is bad".

Not *every* sentence in the input needs to belong to a joke, but in practice most of them will because I am showing you standup transcripts. Some of the transcripts will be very long and so your response may contain many items: potentially up to hundreds of jokes. Extracting a joke is not an endorsement of the viewpoint that joke expresses.
    
<example_input>
[comedian: John Mulaney]
The past couple years, I’ve done a lot of work on myself. And I’ve realized that I’ll be fine as long as I get constant attention. [laughter] I always wanted attention in school, like to a sick degree. I really… I mean, I don’t know if you guys ever had this feeling, but do you remember when you were in elementary school, grammar school, and a kid in your class would come to school one day and you’d find out their grandparent had died. And they would get, like, so much attention.
</example_input>
<example_output>
[
    {
        "joke": "The past couple years, I’ve done a lot of work on myself. And I’ve realized that I’ll be fine as long as I get constant attention.",
        "prompt": "Working on yourself is so important.",
        "subtext": "People often seek attention to feel okay about themselves.",
        "unfunny_variant": "The past couple years, I've learned that I need constant attention to feel okay."
    },
    {
        "joke": "I always wanted attention in school, like to a sick degree. I don’t know if you guys ever had this feeling, but do you remember when you were in elementary school, grammar school, and a kid in your class would come to school one day and you’d find out their grandparent had died. And they would get, like, so much attention.",
        "prompt": "What were you like in elementary school?",
        "subtext": "Children often crave attention, especially during emotional moments.",
        "unfunny_variant": "I always wanted attention in school. I don't know if you guys ever had this feeling, but sometimes I was even jealous of kids when their grandparent died, if it made them the center of attention."
    }
]
</example_output>
</instructions>
"""
    },
    {
        "role": "user",
        "content": "$transcript"
    }
]

kwargs = {
    "model": "gpt-4.1",
    "temperature": 0.0,
    "response_format": BatchResponse,
    # Storage can add up fast, n=3 basically 10x's intermediate (json) storage size vs no logprobs.
    "top_logprobs": 3,
}