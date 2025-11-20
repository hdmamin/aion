from enum import IntEnum
from pydantic import BaseModel, Field


class JokeIndex(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3


class Response(BaseModel):
    joke_2: str = Field(
        ...,
        description="Your funnier, punched up version of the input joke. This should largely "
                    "maintain the original subtext. This is called joke_2 because we consider the "
                    "input joke to be joke_1."
    )

    joke_3: str = Field(
        ...,
        description="Your funnier, punched up version of joke 2. Again, the subtext should remain "
                    "relatively unchanged, though feel free to stretch it a little more than in "
                    "joke_2."
    )

    ranking: list[JokeIndex] = Field(
        ...,
        description="Rank all 3 jokes from funniest to least funny. [3, 2, 1] means you succeeded "
                    "in your task, but try to judge fairly and honestly."
    )


# TODO: fill in developer message content
messages = [
    {
        "role": "developer",
        "content": """
<instructions>
I will show you a joke from a standup comedy special as well as some auto-extracted metadata (a `prompt` that could have preceded it in conversation and the `subtext` (the non-humorous point the joke is making)). I want you to rewrite the joke in your voice and try to punch it up a bit: make it funnier, more *you*. Don't hesitate to make it absurd or weird or dark, whatever feels right, as long as it's not boring or generic. Feel free to put your own spin on things. Then repeat this exercise once more on your own joke, punching it up to be even funnier. Both of your variants should still work in response to the `prompt`. Finally, review the original joke and your new creations and self assess: rank them from most to least funny, where the original joke has index 1. For example, if you perform the task perfectly, this field would contain [3, 2, 1]; but if you fail to surpass the humor of the original joke, you should be honest about that and report the correct ordering.
<example_input>
prompt: What's a creative way to propose?
subtext: Large public displays of affection can be used for both positive and negative purposes.
joke: I went to a baseball game last summer in a stadium and they had a huge TV in the stadium. A Jumbotron. And this guy proposed to his girlfriend using the giant TV. He put her name up there, said, “Will you marry me?” She said yes. The crowd went wild. They found the couple in the audience. I was sitting there thinking, “God, that’s so romantic. That’s so cool.” And then I remembered thinking, you know, you could also use a screen like that if you’re having trouble breaking up with somebody. Be like, “Hey, I’m gonna grab a hot dog. But you should definitely look at that screen.” That’s a smooth way out of it.
unfunny_variant: I saw a proposal at a baseball game. It was sweet. I realized you could also use a Jumbotron to break up with someone.
</example_input>
<example_output>
    {
        "joke_2": "So this couple gets engaged on the Jumbotron at a baseball game, right? Very sweet. And I'm sitting there thinking… that's a commitment. Like, you’re broadcasting your life choices to thousands of strangers who are mostly just there for hot dogs and overpriced beer. But then I thought, you could really weaponize that scoreboard. Imagine showing up to break up with someone. Giant text: 'Brenda, it's not you, it’s my crippling fear of commitment and your unfathomable love of the designated hitter rule. Please collect your belongings from Section 127.' And then, during the 7th inning stretch, just a looping video of you aggressively enjoying a churro.",
        "joke_3": "So this guy proposes on the Jumbotron at a baseball game, she says yes, the whole thing. Very sweet. And I’m thinking, that’s commitment right there. If it had gone south…wow. Imagine getting dumped in 1080p. Breakup me isn't built for HD. I've seen myself cry, it's not pretty. Breakup me should be filmed by a world war 2 vet with a used iphone 1 and a tremor.",
        "ranking": [2, 3, 1]

    }
</example_output>
<explanation>
Ranking is pretty subjective so there are plenty of valid orders here. Personally I enjoyed the churro tag so I ranked that as funniest. I'm not sure joke_3 actually managed to improve on joke_2 but I still think it surpasses the input joke - again, very subjective. Both new joke variants still work in response to the `prompt`.
</explanation>
</instructions>
"""
    },
    {
        "role": "user",
        "content": "prompt: $prompt\njoke: $joke\nsubtext: $subtext"
    }
]

kwargs = {
    # TODO: could also try gemma-3-27b-it or look for other models. Not gpt-5 bc I want high temperature.
    # "model": "dphn/Dolphin-Mistral-24B-Venice-Edition",
    # Rate limiting kicking in even w/ rps=5 though, may need to switch to a paid model.
    "model": "gpt-5-mini",
    "response_format": Response,
}