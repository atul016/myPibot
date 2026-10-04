"""send_photo's muscle: openbot-chat sends the camera frame it took for this turn, captioned with the reply."""


def run(ctx: dict) -> str:
    if not ctx.get("photo"):
        return "You couldn't send a photo: your camera is off."
    ctx["send_photo"] = True
    return "You're sending them the attached camera photo, with your reply as its caption -- say what's in it."
