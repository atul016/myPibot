"""The texts openbot-chat starts on its own, dry-run with a fake WhatsApp client
(no LLM, isolated state): the evening summary goes out once; while someone asked
for a pause (skills/pause_texting) the summary, the dream and the mind's texts
wait -- but a reminder they asked for and an urgent battery warning still go.

    cd ~/openbot && python3 -m tests.test_chat_sending
"""
from tests._audio import isolate_state

isolate_state()

import time  # noqa: E402

import services.chat as chat  # noqa: E402
from common import outcomes, state  # noqa: E402


class FakeClient:
    def __init__(self):
        self.sent = []

    class _Sent:
        def __init__(self, n):
            self.ID = f"m{n}"

    def send_message(self, jid, text):
        self.sent.append(("text", jid.User, text))
        return self._Sent(len(self.sent))

    def send_image(self, jid, photo, caption):
        self.sent.append(("photo", jid.User, caption))
        return self._Sent(len(self.sent))


outcomes.record = lambda *a, **k: None
chat._ask = lambda persona, Reply, turn, query, history, image: (None, "My day was good.", '{"reply": "My day was good."}')
me = "15550100"
client, histories = FakeClient(), {}

chat._tell(client, None, None, histories, "summary", "It's evening...", "fallback", [me])
assert client.sent == [("text", me, "My day was good.")], client.sent  # one summary, in its words
assert histories[me][-1]["content"] == '{"tone_action": "none", "reply": "My day was good."}'  # its answer has context

state.update_session({"texts_paused_until": {me: time.time() + 3600}})
for kind in ("summary", "dream", "jev"):
    chat._send_all(client, histories, f"a {kind} text", None, kind, numbers=[me])
assert len(client.sent) == 1, client.sent  # paused: none of those went
chat._send_all(client, histories, "Reminder: call mom", None, "reminder", numbers=[me])
chat._send_all(client, histories, "My battery is at 4%", None, "urgent", numbers=[me])
assert [t for _, _, t in client.sent[1:]] == ["Reminder: call mom", "My battery is at 4%"], client.sent

state.update_session({"texts_paused_until": {"*": time.time() + 3600}})  # said out loud: everyone
chat._send_all(client, histories, "late summary", None, "summary", numbers=[me])
assert len(client.sent) == 3
state.update_session({"texts_paused_until": {me: time.time() - 1}})  # the hour's over
chat._send_all(client, histories, "late summary", None, "summary", numbers=[me])
assert client.sent[-1][2] == "late summary"
print("test_chat_sending: ok")
