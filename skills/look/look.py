"""look's muscle. Out loud: a head turn (movement/actions.LOOK_ANGLES) -- openbot-alive's face
tracking may take it back. The mind's tool: turn, see (common/vision.py), turn back, say what's there."""
import config as cfg
from common import journal, policy, vision
from common.motor_client import dispatch


def run(ctx: dict, direction: str) -> str:
    if ctx["channel"] != "mind":
        if not dispatch([f"look {direction}"], wait=True):
            return "You couldn't turn your head just now: you're busy driving, or your body isn't answering."
        journal.log("did", f"looked {direction}")
        return f"You turned your head to look {direction}."
    note, turned = "", direction != "ahead" and policy.evaluate("presence").allowed
    if direction != "ahead" and not turned:
        direction, note = "ahead", " (couldn't turn your head right now, so you looked straight ahead)"
    if turned and not dispatch([f"look {direction}"], wait=True):
        turned, direction, note = False, "ahead", " (your head didn't turn, so you looked straight ahead)"
    try:
        seen = vision.look(cfg.LLM_BASE_URL, cfg.LLM_MODEL, direction,
                           head=cfg.LOOK_ANGLES.get(direction) if turned else None)
    finally:
        if turned:
            dispatch(["look ahead"], wait=True)
    if seen is None:
        return f"look {direction}: your camera didn't work this time{note}"
    change = f" -- changed since last time: {seen['what_changed']}" if seen["changed"] else ""
    found = f" -- {seen['bearings']}" if seen.get("bearings") else ""
    if found:
        journal.log("found", seen["bearings"])
    return f"look {direction}: {seen['scene']}{change}{note}{found}"
