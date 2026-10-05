"""speak's muscle: the mind's own words, out loud (its say(): spoken, journaled, then it listens for an answer)."""


def run(ctx: dict, text: str) -> None:
    ctx["say"](text)
