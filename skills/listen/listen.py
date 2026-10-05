"""listen's muscle (a tool): a few seconds of the room from openbot-ears -- what it sounded like, and how loud."""
import time

from common import hearing, surprise

LISTEN_S = 5


def run(ctx: dict) -> str:
    time.sleep(LISTEN_S)
    heard = hearing.read()
    if heard is None or not heard["levels"]:
        return "listen: your hearing service (openbot-ears) isn't answering right now -- that's not silence"
    levels = heard["levels"]
    recent, room = levels[-LISTEN_S:], sorted(levels)[len(levels) // 2]
    return (f"listen: {hearing.describe(heard['sounds'][-LISTEN_S:])}; loudness over the last {LISTEN_S}s {recent} "
            f"(typical for this room ~{room}; a bang {surprise.LOUD_FLOOR:.0f}+)")
