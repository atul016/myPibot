"""introspect's muscle: how the body is, out loud -- composed by code (ctx["status"]), never made up by the LLM."""


def run(ctx: dict) -> None:
    ctx["say"](ctx["status"]())
