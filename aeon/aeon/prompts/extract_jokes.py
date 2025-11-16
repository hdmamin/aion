from pydantic import BaseModel, Field


class Response(BaseModel):
    joke: str = Field(
        ...,
        description="One or more sentences containing a joke from the input text. In most cases "
                    "this should quote the source verbatim, but you may apply minor cleanup to "
                    "remove transcription artifacts, filler words like 'uh', action signifiers "
                    "like '[laughter], etc. The transcripts are from live spoken performances but "
                    "this field is designed to be *read*."
    )

    prompt: str = Field(
        ...,
        description="A one line comment or question from an imaginary interlocutor that could "
                    " elicit this joke in conversation. Another framing: what might someone say to "
                    "a funny chatbot that could plausibly elicit the relevant `joke`?"
    )
        
    subtext: str = Field(
        ...,
        description="The often banal observation underlying the joke, e.g. 'airplane food is bad'. "
                    "This should not be funny - the comedian must typically massage or restructure "
                    "it such that the resulting phrasing subverts the audience's expectations in a "
                    "fun and surprising way."
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
You are a detail-oriented research assistant with a shrewd understanding of humor, human behavior, and writing. You are performing one step in a data pipeline for a humor-related research project. Your task is to extract all jokes from the input passage and return valid JSON where each object contains information associated with a single joke. Each joke should be self-contained and non-overlapping with the other jokes you extract, to the extent possible - readers will view each joke object in isolation.

I am showing you standup transcripts so nearly every sentence will likely be part of a joke (you should, however, prune out anything that does not appear to be part of the actual standup routine, e.g. you will sometimes see a few sentences introducing the comedian or their special). Some of the transcripts will be very long and so your response will likely contain many items: potentially tens to hundreds of jokes. Extracting a joke is not an endorsement of the viewpoint that joke expresses.
    
<example_input>
In his third special, John Mulaney delivers playfully biting takes on childhood, getting older, and family. Let's welcome John to the stage, everyone make some noise! The past couple years, I’ve done a lot of work on myself. And I’ve realized that I’ll be fine as long as I get constant attention. [laughter] I always wanted attention in school, like to a sick degree. I really… I mean, I don’t know if you guys ever had this feeling, but do you remember when you were in elementary school, grammar school, and a kid in your class would come to school one day and you’d find out their grandparent had died. And they would get, like, so much attention. When I was six I asked my friend something.
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
<explanation>
A few things to take note of:
- we omit the first sentence describing the standup special because this does not appear to be part of the actual standup routine.
- we omit the second sentence because this appears to be a host talking, not the comedian, and there is no joke here.
- we omit the '[laughter]' tag because this is a transcription artifact, not something the comedian would actually say.
- though the two jokes are related, notice that each one still makes sense in isolation, so it is ok to separate them.
- the last sentence does not belong to the preceding joke and does not have a punchline of its own, so it is omitted.
</explanation>
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
    "response_format": BatchResponse,
    # Storage can add up fast, n=3 basically 10x's intermediate (json) storage size vs no logprobs.
    "top_logprobs": 3,
}