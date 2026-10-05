"""time_check's muscle: the time, out loud -- from the clock, not the LLM."""
import datetime


def run(ctx: dict) -> None:
    ctx["say"](f"It is currently {datetime.datetime.now().strftime('%I:%M %p').lstrip('0')}.")
