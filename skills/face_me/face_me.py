"""face_me's muscle: which way their voice came from while they said it (the reSpeaker array, via
openbot-ears: hearing.voice_bearing over the utterance's own window, ctx["heard"]), then a turn by that
much where it stands, measured by the IMU (navigate.turn_by in openbot-alive, which says how it went)."""
from common import hearing, journal
from common.motor_client import navigate

ALREADY_DEG = 20  # about this close to straight ahead: already facing them


def run(ctx: dict) -> str:
    bearing = hearing.voice_bearing(*ctx.get("heard", (0.0, 0.0)))
    if bearing is None:
        return "You couldn't tell which way their voice came from -- ask them to come where you can see them."
    if abs(bearing) < ALREADY_DEG:
        return "You're already facing them."
    ok, err = navigate({"turn": bearing})
    if not ok:
        return ("You can't turn now: you're already driving somewhere." if "already" in err
                else "You can't turn right now: your wheels aren't answering.")
    side = "right" if bearing > 0 else "left"
    journal.log("did", f"turning about {abs(bearing):.0f} degrees {side} to face the voice")
    return f"You're turning to face them: their voice came from about {abs(bearing):.0f} degrees to your {side}."
