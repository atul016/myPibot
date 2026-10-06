# OpenBot

A small robot that lives on your desk: you talk to it, it talks back, it
looks around with a camera, remembers people and what happened, and has a
quiet "inner life" -- it notices things and sometimes speaks up on its own.
Its brain is **a local LLM** (any OpenAI-compatible server: Ollama, MLX
Serve, llama.cpp...), so nothing goes to a cloud.

**You don't need a robot car.** OpenBot runs on any Linux computer (a
Raspberry Pi is ideal) with a microphone and a speaker. A camera is
optional. With a [SunFounder PiCar-X](https://docs.sunfounder.com/projects/picar-x-v20/en/latest/)
it also gestures, drives to things ("go to the pink toy"), follows you
("follow me"), explores, and backs away from table edges. Other robots can be added (see
[Adding your own robot](#adding-your-own-robot)).

The bot's identity is a **persona**: name, wake word, personality and
voice. **Rocky** (from *Project Hail Mary*, with an alien chord voice) is the
example in `personas/rocky/`; make your own in a few minutes.

## Quick start

You need:
- **Linux with Python 3.12+**: Raspberry Pi OS Trixie / Debian 13 (tested), or Ubuntu 24.04 (untested). Recording uses `arecord`, so not macOS/Windows. Tested on a headless Pi 5; on a desktop with PulseAudio/PipeWire the speaker can be "busy".
- A **USB microphone** and a **speaker**. Optional: a camera (Pi camera or USB webcam).
- An **LLM server** that can see images and give JSON output. Easiest:
  [Ollama](https://ollama.com) with a vision model (`ollama pull qwen2.5vl:7b`). A Pi
  is too slow to run a good one itself, so run it on a desktop/laptop on the same network.

```bash
sudo apt install -y git
git clone https://github.com/atul016/myPibot.git ~/openbot && cd ~/openbot
./setup.sh                  # or: ./setup.sh --picarx   on a PiCar-X
nano openbot.env            # LLM address + model, mic name (arecord -l), speaker volume control
sudo systemd/install.sh     # installs and starts the services
```

Then say **"Rocky"** (the persona's name) and talk. Say "be quiet" to end the
conversation, "go to sleep" to turn everything off until "Rocky, wake up".
The dashboard is at `http://<robot-ip>:8080`; logs: `journalctl -u openbot-wake-listen -f`.

**Settings** live in `openbot.env` (template: `openbot.env.example`; every
`OPENBOT_*` setting is in `config.py`). The important ones:

| Setting | What |
|---|---|
| `OPENBOT_BODY` | `none` (talks, sees, thinks -- never moves) or `picarx` |
| `OPENBOT_PERSONA` | which `personas/<name>/` to be |
| `OPENBOT_LLM_BASE_URL`, `OPENBOT_LLM_MODEL` | the brain |
| `OPENBOT_STT_DEVICE` | part of your mic's name in `arecord -l` (openbot-ears opens it) |
| `OPENBOT_DOA_FRONT` | with a reSpeaker XVF3800: the array's angle that points at the robot's front -- talk from straight in front while `python3 -m services.ears --doa` runs (unset: no voice direction); from its right the number should go up, else set `OPENBOT_DOA_CCW=1` |
| `OPENBOT_IMU_AXES` | with a DFRobot 6 DOF IMU on I2C (0x4A): its axes that point forward, right and down, as mounted (default `+x,+y,+z`: flat, X arrow forward) -- lift the front an inch and see which one moves |
| `OPENBOT_LOUD_FLOOR` | how loud a sudden sound must be (default 1500, for a USB dongle; a reSpeaker XVF3800 needs ~5000 -- talk and clap, then read `127.0.0.1:9001/hearing`) |
| `OPENBOT_MIXER` | `card:control` for "louder"/"softer" (`amixer -c 0 scontrols`) |
| `OPENBOT_CAMERA_CMD` | a USB webcam instead of the Pi camera (example in the template) |

Editing on a laptop and running on the robot? `./deploy.sh user@robot` copies
the folder over, then run `sudo ~/openbot/systemd/install.sh` there.

## Make your own persona

1. `cp -r personas/rocky personas/<name>` -- the folder name is the bot's name, lowercase.
2. Edit `persona.py`: `NAME`, `WAKE_WORDS`, `piper_voice` (any
   [Piper voice](https://huggingface.co/rhasspy/piper-voices), downloaded on first use).
   Delete `speak_overlay` and `transform` unless you want Rocky's alien voice and grammar.
3. Edit `prompt.py`: who it is, how it talks.
4. Set `OPENBOT_PERSONA=<name>` in `openbot.env`, and re-run `sudo systemd/install.sh`.
   Memories and the journal are kept: every persona shares them.

**Pick a plain, common English word as the name.** The wake word is heard by a
small offline speech model that only knows common words -- an unusual name
will never wake it.

## Adding your own robot

Everything that moves goes through one Unix socket, `/tmp/openbot-alive.sock`
(`common/motor_client.py`). For a PiCar-X, `services/alive.py` answers it.
For another robot, write a small service that answers the same JSON, one
request per connection:

| Request | Reply | Meaning |
|---|---|---|
| `{"actions": ["nod", "look left"], "wait": true}` | `{"ok": true}` | do these gestures in order (`wait`: reply when done) |
| `{"navigate": {"approach": "pink toy"}}` / `{"navigate": {"follow": true}}` / `{"navigate": {"explore": 60}}` | `{"ok": true}` or `{"ok": false, "error": "..."}` | start a drive |
| `{"navigate_cancel": true}` | `{"ok": true, "was_driving": bool}` | stop driving now |

Then, in `config.py`, add your body name next to `picarx`: its gesture names
(the LLM picks from them to match its mood), `look <direction>` head turns if
it has a moving camera, and `CAN_DRIVE`. Publishing sensor readings to
`state/sensors.json` (`common/sensors.py`) lets the mind notice someone
approaching. The driving logic (`movement/navigate.py`) is written against a
small `Body` interface and tested in a simulator, so a wheeled robot can reuse
it by implementing that interface.

## How it works

Seven small systemd services (eight with WhatsApp) instead of one process, so a crash or hang in
one concern can't take the others down, and a background "inner life" loop
that can act without being addressed -- speaking, gesturing, or just staying
quiet -- alongside the reactive wake-word assistant. The design notes below
come from building it on a PiCar-X ("Walle"); hardware-specific lessons are
marked as such.

Design history: this replaces an earlier "desk-bot", a single reactive
process built on `sunfounder_voice_assistant.VoiceAssistant`. That version's hardware-specific lessons (cliff/proximity
thresholds, the sudo/speaker-amp requirement, the multi-turn session
design, the chord-overlay TTS approach) carry forward here as facts, not as
copied code -- see each module's docstring for what changed and why.

### Architecture

```
openbot-alive (user)         openbot-wake-listen (user)      openbot-mind (user)
  owns Picarx() PERMANENTLY    wake word -> STT -> reactive     awareness -> reflection
  cliff reflex, near-latches   voice-loop turn                  -> expression (autonomous)
  head follows faces (gaze)           |                                |
        ^                             |                                |
        +--- common/motor_client.py: {"actions": [...], "wait": bool} -+
        |         (Unix socket -- request a gesture, don't build one)
        |                              |                                |
        +----------------- common/state.py, events.py, sensors.py ------+
                  (FileLock-guarded JSON in state/ -- the only
                   coordination point between these processes)
                                       |
                              openbot-speak (root)
                          the ONLY process touching the amp/audio device
                                       |
          openbot-tasks (user): says the reminders people asked for, when due
                                       |
                            openbot-dashboard (user)
                       reads state/*.json, no hardware access at all
```

With `OPENBOT_BODY=none` there is no `openbot-alive`: gesture and drive
requests simply fail (quietly), and the prompts tell the LLM it can't move.

On a PiCar-X, `openbot-alive` is the *permanent* sole owner of the `Picarx()` handle --
confirmed live that a second, independent `Picarx()` in another process
does not work on this hardware (`lgpio.error: 'GPIO busy'`): lgpio's GPIO
line claims are exclusive per open chip handle, and stopping motors/
ActionFlow does not release them. An earlier design tried a lease/yield
hand-off (`openbot-alive` releasing the handle on request, the requester
building its own); that's what hit the error. `wake_listen.py`/`mind.py`
now ask `openbot-alive` to perform an action over a socket instead --
exactly the same privilege-boundary shape already used for audio.

## The four ideas borrowed from SPARK (adrianwedd/spark), adapted here

Reviewed as prior art for a PiCar-X voice assistant with a genuinely more
modular shape. Four specific pieces, all fully built (not stubbed):

1. **A whitelist + param validation for the one truly free-form LLM
   decision point** -- `services/mind.py`'s `validate_expression()`, which
   holds the pick to that skill's own spec (`common/skills.py` `check()`). The
   *reactive* turn (`services/wake_listen.py`) doesn't need this: Ollama's
   `format=<schema>` grammar-constrained output already makes the model
   structurally unable to emit anything outside `{reply, tone_action}`, and
   what it does is its skills, decided by a separate call before the reply
   (see Skills) -- the reply is told what was done. The autonomous mind's
   `{action, params}` choice is a genuinely free pick among a small set,
   closer to SPARK's tool-call shape -- so it gets the same defense.
2. **An STT fallback cascade** (`common/stt.py`) -- faster-whisper first
   (better short-utterance/accent accuracy), Vosk as a working fallback,
   both with the same anti-hallucination filters (phantom-phrase rejection,
   repetition/word-rate checks).
3. **`arecord`-based mic capture** (`common/mic_stream.py`) instead of
   PortAudio (PyAudio/sounddevice), which has been observed silently
   dropping a large fraction of samples on this USB mic class -- invisible
   on every offline metric, only caught by a live loopback check.
4. **Coordinating who drives the car across three processes** -- adapted
   from SPARK's GPIO lease, but ended up shaped differently once tested on
   real hardware: a lease/hand-off model (yield the Picarx handle, let the
   requester build its own) hit `lgpio.error: 'GPIO busy'` live, because
   this hardware's GPIO library doesn't release line claims just because
   motors stopped. `openbot-alive` owns `Picarx()` permanently instead, and
   `wake_listen.py`/`mind.py` request actions over a socket
   (`common/motor_client.py`) -- the same shape as `openbot-speak` for
   audio, not a lease at all. See the Architecture section above.

Also adopted: per-service health with staleness (`common/health.py`,
replacing a single watchdog heartbeat with "which service died"), and a
single audio-output chokepoint (`common/policy.py`, gating every spoken/
expressive action through one asleep/motion-confirm check instead of
trusting each caller to remember).

Explicitly **not** adopted (SPARK-specific, not applicable here): Ollama
Cloud, a Cloudflare tunnel, Bluesky/blog/self-evolution daemons, Home
Assistant integration, Google Find Hub, jailbroken personas, PIN-based REST
auth, and its ~1235-test suite.

## Feeling alive: surprise, eyes, curiosity, conversation

- **Wake on surprise** (`common/surprise.py`, `services/mind.py`) -- mind samples
  sensors + what openbot-ears heard every 2s. Something approaching, a sudden sound, being
  picked up, or a changed camera scene triggers a reflection *right away*
  (rate-limited by `SURPRISE_MIN_GAP_S`), not just on the 5-min idle timer.
  The robot's own speech/gestures stamp `self_noise_ts`; sound/vision
  surprises are ignored for `SELF_NOISE_QUIET_S` after, so it doesn't react
  to itself.
- **Camera** (`services/camera.py`, `openbot-camera`) -- the camera's sole owner: `rpicam-vid`
  MJPEG at ~10fps (NoIR tuning), served on :9000 (`/mjpg`, `/snapshot.jpg`) and relayed live
  in the dashboard's Camera tab, with the object detector's boxes drawn over it. Everyone else
  asks it for frames. The detector (`common/objects.py`) runs every 2s, or on every frame
  (~9/s on a Pi 5) while someone calls `objects.want_fast()` -- the Camera tab, or following.
- **Ears** (`services/ears.py`, `openbot-ears`) -- the microphone's sole owner (`arecord`), served
  on 127.0.0.1:9001 only -- it's the house's live audio: `/pcm` is the live stream (openbot-wake-listen
  listens to it, ~3ms behind the mic), `/hearing` each of the last 30s's peak loudness and what it
  sounded like (YAMNet on every second; "own voice" while the robot talks). The mind asks it
  (`common/hearing.py`) -- no file in between, so "not answering" can't pass for "silence".
- **Eyes** (`common/vision.py`) -- a frame from openbot-camera (fallback: one-shot `rpicam-still`; never vilib) every
  `VISION_INTERVAL_S` -> the vision LLM describes the scene and says what
  changed since the last look that way. Frames darker than `DARK_BRIGHTNESS`
  skip the LLM ("too dark to see") -- that's also how "the lights came on"
  becomes a surprise. The latest scene is injected into conversation prompts.
- **Curiosity + tools** -- each reflection may chain up to `MIND_MAX_STEPS`
  tool calls (`look` left/right/up/ahead via the camera gimbal, `listen`,
  `recall`) before acting. A surprise arriving mid-chain is handed to the
  next step. The only wheel moves the mind may choose are the fist bump and
  bullfight nudges (short, cliff-checked); driving anywhere is a spoken command.
- **Its own drives** (`services/mind.py`) -- besides speaking and gesturing, a reflection may:
  `drive_to` / `explore` (drive to something it sees, or wander -- the same cliff-guarded
  `movement/navigate.py` as spoken commands; never in the dark), `sleep` on its own when
  it's dark and quiet with nobody around (or the battery is very low), and `wish` for
  something it can't do (written to `state/mind/self/wishes.md` -- read it). Its awareness
  includes code-derived **needs** (battery, dark room, camera trouble), the **people
  routines** the nightly pass learned, when each person was last seen, the goal to pursue
  now, and `self/what-works.md`: 3-6 rules it rewrites every night from how people reacted
  to what it did unprompted.
- **It starts conversations** -- after the mind speaks, wake-listen listens ~20s for an
  answer without the wake word; an answer opens a normal session, seeded with what the
  mind said and where the last conversation (today, yesterday, this week) left off.
- **Knows which way it's looking** (`services/alive.py` -> `sensors.json` `head`, `common/vision.py`) --
  the head follows faces and glances about, and people pick the robot up and turn it; without knowing
  that, it wrote "the room layout shifted again without me moving" 55 times in a day. Now a look is
  compared only with the last one from about the same head angle (20-degree steps), motion right after
  its own head moved (or while it's carried) isn't news, and being picked up, set down or driving clears
  its views and tells it "you may face another way now".
- **Names what it sees and hears** -- objects (`common/objects.py`: NanoDet, OpenCV's model zoo,
  every 2 s in openbot-camera: "a person, a chair and a laptop" in every prompt, and grounding each
  look), sounds (`common/sounds.py`: Google's YAMNet on every second openbot-ears hears: "a sudden
  sound -- it sounded like: Knock", and the `listen` tool: "it sounded like: Speech, Music"), and voices (`common/voices.py`: WeSpeaker voice prints, learned
  like faces -- while it sees exactly one face it knows, that person's words teach it their voice;
  when no face says who's talking, the voice can). Models: `setup.sh`, into `~/.openbot-models`.
- **Motion** (`services/camera.py`) -- frame-to-frame change, twice a second, becomes a
  "something is moving" surprise when sustained (one jump is the head itself turning).
- **Emotes and a dance** (`movement/actions.py`, after [SPARK](https://github.com/adrianwedd/spark)'s
  `px-emote`/`px-dance`) -- eased head poses `curious`, `happy`, `excited`, `sad`, `shy` join the
  preset gestures the LLM picks from; `dance` (a circle, then a figure 8) on "dance for me", and
  as the mind's own choice only with `OPENBOT_ON_FLOOR=1` (an arc meets a table edge at any angle).
- **Goals, reminders, watches, rest** (`common/agenda.py`) -- up to 3 open
  goals in `state/mind/agenda.md`; resolving one saves "question ->
  conclusion" as a lesson. `remind_me` wakes it later (and rests until
  then), `watch` flags a kind of surprise it cares about, `rest` skips idle
  reflections -- so a dark, empty room doesn't produce a "staring into the
  void" thought every 5 minutes.

## Faster ears: Whisper on the LLM's machine (optional)

Speech-to-text can run on the LLM machine's GPU (whisper.cpp `whisper-server`,
large-v3-turbo): ~0.25s per utterance and far more accurate than the Pi's own
`base.en` (~2.4s), which stays as the automatic fallback when that machine is off.
On a Mac: `tools/setup-mac-whisper.sh <robot-ip>` (a launchd agent on :8178;
the script's header shows how to remove it), then `OPENBOT_MAC_WHISPER_PORT=8178`.
The robot finds it on the same host as the LLM server. End of speech -> transcript:
~1.2s (was ~3.2s on the Pi alone).

## Text it on WhatsApp (optional)

`openbot-chat` (`services/chat.py`) lets you text the bot. It answers as itself: the same
prompt, journal and memories as when you talk to it, plus what its camera sees right now,
and it sends you that photo when you ask to see ("show me"). Send it a photo and it looks
at yours instead. What it can do from a text is its [skills](#skills-what-it-can-do): it
decides from your words -- no exact commands. It never moves from a text: nobody may be
watching the table edge, so it tells you to ask out loud.

The robot logs in as a **linked device of a spare WhatsApp number**, the way WhatsApp Web
does ([neonize](https://github.com/krypton-byte/neonize)). It only connects out, so no
public address or tunnel is needed. But it's unofficial: it breaks WhatsApp's terms, and a
ban is permanent. **Use a number you can afford to lose, never your own.**

1. Put the spare number's SIM in a phone and set up WhatsApp on it.
2. In `openbot.env`, set who may text it (country code first; everyone else is ignored):
   `OPENBOT_CHAT_ALLOW="+91 98765 43210"`. Then `sudo systemd/install.sh`.
3. `journalctl -u openbot-chat -f -o cat -a` shows a QR code. On the spare phone:
   WhatsApp > Settings > Linked devices > Link a device, and scan it.
4. Text the spare number from an allowed phone.

Open WhatsApp on the spare phone at least every 14 days, or WhatsApp logs the robot out
(the dashboard shows `openbot-chat` stale; step 3 again). The login is kept in
`~/.openbot-whatsapp/`. It ignores voice notes, videos, stickers and group chats.

**Who's texting** is learned like a face: text "My name is Atul" once (or "this is atul", for
someone it already knows) and that number is Atul's -- the person it knows by face and in its
notes, which gain "Atul texts me on WhatsApp". The number itself stays in `~/.openbot-whatsapp/`.

**Asking it to do things** -- in your own words: "stop texting me for an hour", "remind me at 17:30
to call mom", "add milk to my list", "send me a pic", "you're too loud", "go to sleep" / "wake up",
"be quiet" (ends a conversation someone is having with it out loud). See [Skills](#skills-what-it-can-do).

**It texts first**, too:
- how its day went, at 8 PM (`OPENBOT_CHAT_SUMMARY_AT=20:00`), and last night's dream in the
  morning (`OPENBOT_CHAT_DREAM_AT=08:00`);
- whenever something happens or crosses its mind that's worth a text -- but neither the LLM nor
  a rule decides that: after each thought it asks [Jev](https://docs.typesafe.ai) (TypeSafe's
  decision model) whether this moment is one you want a text about (`OPENBOT_CHAT_TEXT_WHEN`,
  default: its mood changes to curious or bored, or something unusually big happens), with its
  mood before and now, the thought, what just happened and when it last texted. On a yes the
  LLM writes the text (maybe with a photo). At most one every 10 minutes, however keen Jev is. Needs a TypeSafe API key on the
  robot -- without one it never texts first (except the summary and the dream):

  ```bash
  ssh -t user@robot 'read -rsp "Jev API key: " k; echo; printf "OPENBOT_JEV_API_KEY=%s\n" "$k" | sudo tee /etc/openbot/jev.env >/dev/null; sudo chmod 600 /etc/openbot/jev.env; sudo systemctl restart openbot-mind'
  ```

  The dashboard's **Jev** tab shows every question asked of Jev -- the whole state sent, the
  options, Jev's whole answer -- newest first. All of them are kept, one file a day, in
  `state/jev/<day>.jsonl` (never trimmed): state in, Jev's choice and probabilities out --
  training data for a classifier of your own that does Jev's job.

## Driving (PiCar-X): designed in a simulator first (`movement/navigate.py`, `sim/`)

"Go to the pink toy" (`navigate.approach`) and "explore" (`navigate.explore`)
are written against a small `Body` interface, and proven in a 2D tabletop
simulator (`sim/world.py`: PiCar-X steering, the ~30-degree ultrasonic cone
with -2 glitches, the 3-channel floor sensor, the camera's 54-degree view
with occlusion) across `sim/scenarios/*.json` x 25 seeds -- before the real
robot. Idea borrowed from lucascosolo/picarx-training; code our own.

```bash
python3 -m tests.test_navigation_sim      # runs anywhere (pure stdlib); pictures in sim/out/
```

What the sim taught (each now a rule in navigate.py):
- **Never reverse blind** -- no rear sensor. Backing up only retraces ground
  just driven forward ("reverse credit"); free reversing fell off the table.
- **The floor sensor is narrower than the wheels** (~5cm vs ~14cm): met at a
  shallow angle, a wheel goes over first. So: no wandering on tables --
  explore stops at the first edge and looks around instead; approaches are
  head-on, and a target past an edge is refused.
- **The ultrasonic is a narrow cone**: something 6cm off-centre leaves it before
  a 14cm stop triggers. Cruise with a 30cm stop; 14cm only for the final approach.
- **An ultrasonic reading isn't the target**: "arrived" also needs the target to look close.

**Driving smoothly** (`approach`): the vision model (~0.5-2s per look) finds the
target once while standing still. After that, an OpenCV tracker (CSRT, started
from the model's box) follows it at ~10Hz while the wheels keep turning and the
steering follows it. The floor and the ultrasonic are checked every 50ms. A
cliff, an obstacle or a lost target stops the car. It then makes the same
careful recovery moves as before, and asks the model again. On the robot,
`movement/real_body.py` implements the `Body`, and "drive: ..." lines in the
openbot-alive log show each look, acquire and move.

Measured (25 seeds each): open table, target behind, target near the edge --
25/25 reached, 0 falls, 0 bumps; box in the way 23/25 (rare soft corner clip);
hidden or off-the-table targets refused 25/25; exploring a room floor -- 0 falls,
a soft bump in ~40% of ~3-minute runs (things beside the path). The physical
numbers in `sim/world.py` are calibration knobs to measure on the real car.

**Following** (`navigate.follow`, "follow me"): the object detector's person boxes,
~9/s, with the head tipped up 30 degrees (measured: then box width tracks distance).
Steer toward them, stop ~0.6m away (box width or the ultrasonic), go again when they
walk on. The head only turns while the car stands still -- a detection is ~0.1s old,
so a moving head would put them in the wrong place -- and turns their way when they're
off to the side, so Rocky can turn after them. Lost: look around with the head; still
nothing: say "Where did you go?" and keep looking ~10s. The first floor edge ends it.
In the sim, with a walking person (waypoints, pauses, missed detections): across a
room 25/25, round to his side 25/25, behind him and back 24/25 (0/25 without the
keep-looking), past a desk edge -- stops at the edge 25/25; 0 falls, 0 bumps.

## Skills: what it can do

Everything it can do is a **skill** (`common/skills.py`): a folder in `skills/` with a Markdown file the LLM reads
(what it does, when to use it, when not, what it needs) and, if it needs one, a Python file
that does it. No phrase lists in code, no rules written into prompts -- edit the `.md` to
change how a skill gets used; a new skill is a new folder.

```
skills/pause_texting/pause_texting.md      skills/pause_texting/pause_texting.py
---                                        def run(ctx, minutes):
name: pause_texting                            ...hold back the texts it starts...
description: Stop texting them first...        return "You won't text them first until 15:20."
where: text, voice
params:
  minutes (integer): how long -- 60 is an hour
---
Use when they ask you not to message them for a while: "stop texting me for an hour"...
```

`where` says which channels may use it (`text`, `voice`, `mind`); `needs: body` hides it on a
robot that can't drive. A param can say more than its type: `(integer, 1-720)`, `(string, max 300)`,
`(direction, default ahead)` -- `direction`, `gesture`, `sound`, `memory_kind` and `watch_kind` are
this body's own lists, and a skill whose list is empty (no sounds without a PiCar-X) isn't offered.
What a decision is shown and what its answer is checked against come from that one line. A turn is three steps, like a person: **decide** (a short LLM call that
sees every skill the body has and answers only with the ones the message asks for), **do**
(code keeps what this channel may do -- a text can't drive -- and runs each skill's `run()`,
which says what happened), then **say** (the persona writes the reply, told exactly what was
done -- so it can't claim a move it didn't make). Reflexes stay in code: an instant "stop",
edge and battery safety.

Measured through the real LLM: texting (`tests/test_chat_skills.py`, 23 texts x 3) 69/69, vs
29/33 when one call both chose and replied ("look left" got "Looking left!" every time); speaking
(`tests/test_voice_skills.py`, 32 things said x 3) 96/96. The decide call adds ~0.15s to a spoken reply (1.25s to the
first sentence, was 1.11s) -- the server caches its fixed prompt.

The mind's actions are skills too (`where: mind`: speak, gesture, look, listen, recall, remember,
remind_me, watch, rest, wish, sleep, drive_to, explore...). Its reflection's list, the JSON schema's
choices, Jev's options and the checking of what it picks all come from the `.md` -- a new action is a
new folder, not four edits in `services/mind.py` that have to agree. `tool: yes` means it sees what
the muscle returns and decides again (look, listen, recall); `effect: audio` or `motion` puts it under
the anti-flap cooldown and the sleep gate, `presence` under the gate only. A skill people use too
gives the mind its own line (`mind: Only for the night, when it's dark and quiet...`) in place of the
when-they-ask text. The mind's own restraint stays in its code: never drive in the dark, sleep only when
tired with nobody in view. Through the real LLM, 24 reflections: 23 picked an action that held up
(22/24 with the old hand-written list), with the same mix of looking, listening and resting.

## The body's sense of turning (IMU, optional)

With a DFRobot Gravity 6 DOF IMU on the robot board's I2C pins, `openbot-alive` reads it 25 times a
second (`common/imu.py`) and publishes how far the body has turned, how it's tilted and whether it's
moving. Turned on the spot -- by someone, or by its own move -- it still knows which way it faces (the
camera's view map turns with it, `vision.turn_by`), instead of guessing until it recognizes a view; and
the mind notices being turned ("someone turned you about 90 degrees to your left"), lying tipped,
and tells an edge (nothing touched it) from being picked up. Its own gestures, moves, drives and the
cliff reflex aren't "someone": the action queue says when the body is moving itself. The turn rate is
taken about the vertical (gyro projected on gravity), so holding it at an angle doesn't skew the count,
and the gyro's bias is learned only while it's still. No IMU: everything works as before.

## Which way a voice comes from (reSpeaker XVF3800, optional)

The XVF3800 array works out which direction a speaker is in (its processed DoA: it picks, by speech
energy, among its focused beams -- `DOA_VALUE` sticks to one beam). `openbot-ears` reads it ten times a
second while someone speaks -- not while the robot talks, which the array would hear as a voice from its
own speaker -- and serves it as degrees from the robot's front (`/hearing`, `"voices"`). Then:

- **it turns to whoever talks to it** -- the wake word, or anything said in a conversation, read for the
  time they were actually talking (not the seconds of thinking after): more than 30 degrees round, the
  whole body turns where it stands, quietly (the conversation does the talking); less, just the head.
  Not when a skill that drives or moves the head was chosen (it does its own moving), nor when the one in
  view is the one talking, nor asleep or with motion switched off, nor when the body turned while they
  talked (that direction was heard from somewhere else);
- **in a conversation it keeps them in view** (`openbot-alive`'s face tracker): a face -- or, with none, a
  person the object detector sees (from the floor a standing adult's face is often above the frame), the
  head tipping up toward where it should be; out of view, it keeps looking where they were; more than 35
  degrees round for 1.5 s, the whole body turns to them;
- **"turn towards me"** (skill `face_me`, out loud only: a text has no voice to point at) does it on request.

Every turn is a three-point turn where it stands (`navigate.turn_by`) watching the IMU's heading, each
move guarded like any drive; it stops if a move takes it further off (a wrong sign never spins it round)
or if moves stop turning it (motors off, an IMU that stopped). Proven in the simulator first (`turn_*`
scenarios: 25/25, 5-8 cm from where it stood).

Setup (`setup.sh` does it): `python3-usb`, and a udev rule so the services can read the array without
root (`/etc/udev/rules.d/99-openbot-respeaker.rules`, group `plugdev`). Then set `OPENBOT_DOA_FRONT`.

## Tasks and reminders

Ask by text or out loud, in your own words: "add milk to my list", "what's on my list?", "I bought the
milk" (skills `add_task`, `list_tasks`, `finish_task` -- one household list), "remind me at 17:30 to
call mom", "remind me in 20 minutes to check the oven" (`remind`). A reminder asked for by text is
texted back by `openbot-chat`; one asked for out loud is said out loud at home by `openbot-tasks`, the
second it's due -- asleep or not, like an alarm, and whatever the mind is busy thinking. The mind's own
reminders (`remind_me`) are in the same list (`common/agenda.py`) and come back to it as a thought.
Both lists live in `state/session.json`, changed under its lock (`state.change_session`) -- never in
`state/mind/`, which Basic Memory indexes: a texted reminder carries a phone number. The dashboard's Mind
tab shows them. Not yet: cancelling a reminder by asking, "tomorrow at 9" said before 9 (it's today's),
texting a reminder that was asked for out loud.

## Talking to it

Say the persona's name ("Rocky") to start a conversation. It stays open -- pauses and silence never
end it -- until you ask it to stop. There are no commands to learn: say what you want in your own
words, and its [skills](#skills-what-it-can-do) decide -- "be quiet" or "that's all for now" ends
the conversation (it stays awake in the background, watching, thinking, and may still speak up);
"go to sleep" turns everything off but the voice listener, head down, until **"Rocky, wake up"**
(plain "Rocky" is ignored while asleep; "Rocky, wake up, what time is it" wakes it and answers);
"louder" / "you're too loud" changes the volume; "go to the pink toy", "come here", "follow me",
"explore", "turn left", "dance", "look up" drive and move it (PiCar-X; see Driving) -- out loud only,
never by text. What it can't do from where you asked, it says so.

Three reflexes skip the thinking: "stop" / "wait" while it drives stops the wheels at once (the
decision a second later only says so), talking over it cuts its speech off, and asleep it hears
nothing but "Rocky, wake up". Measured through the real LLM (`tests/test_voice_skills.py`, 32
things said x 3): 96/96 -- including the talk the old phrase lists had to learn to ignore
("welcome back", "my back hurts", "do you like to dance?"). The mind pauses only
while someone's actually talking (`state.conversation_active`: speech in the
last 2 min), so an open-but-idle session doesn't freeze it.

## Memory: one life record + notes by kind (Basic Memory)

Everything lives as plain markdown under `state/mind/` (open it in Obsidian
or any editor), indexed by [Basic Memory](https://github.com/basicmachines-co/basic-memory)
(AGPL-3.0, run unmodified as the `openbot-memory` service on 127.0.0.1:8765):

```
state/mind/
  journal/2026-10-02.md     one continuous record: every heard/said/noticed/thought/found/did line,
                            appended by mind, wake-listen and alive's reflexes (common/journal.py)
  summaries/2026-10-02.md   rolling "today so far", rewritten every SUMMARY_INTERVAL_S
  people/ places/ lessons/ self/   durable notes by kind (common/memory.py)
  agenda.md                 open/done goals
```

- Every prompt (mind and conversation) gets the day's summary + the last
  journal lines; `memory.recall(query)` searches notes AND journal lines
  (semantic + full-text, ~1s) -- the mind uses it each reflection and as a
  tool, conversations use it with your words as the query.
- Once a day (first thing after midnight) a "dream" pass distills
  yesterday's journal into notes by kind.
- Files are written directly with complete frontmatter (incl. `permalink`)
  -- Basic Memory rewrites files missing one, which would race with appends.
- Memory server down -> writes still land on disk; recall returns nothing.

- **Conversation** -- a session keeps chat history (`MAX_HISTORY_MESSAGES`),
  so follow-ups make sense. **Barge-in**: say "stop" / "wait" / "hold on" /
  "be quiet" while Rocky talks to cut it off (`OPENBOT_BARGE_IN=0` to disable;
  only overlay personas like Rocky are interruptible).

## Privilege model

Confirmed on the PiCar-X: motors, sensors, camera, GPIO/I2C need **no**
sudo -- only audio playback does (the speaker stays silent as a normal
user, works instantly under sudo). So only `openbot-speak` runs as root;
everything else runs as the normal deploying user. Other services reach it
over a Unix socket (`common/speak_client.py`), never by running as root
themselves -- one privilege boundary, one process with Piper's voice model
kept loaded (not reloaded per utterance).

## Config vs. persona

- **`config.py`** -- properties of *this physical robot*: the body, LLM base URL/
  model, STT device/language, safety thresholds
  (`SAFE_DISTANCE`/`DANGER_DISTANCE`/`CLIFF_REFERENCE`),
  dashboard port. Env-var overrides follow `OPENBOT_*`.
- **`personas/<name>/`** -- everything about a bot's identity: name, wake/
  sleep words, system prompt, wake-greeting lines, Piper voice, an optional
  text-transform hook (Rocky's broken-grammar speech pattern), an optional
  TTS-overlay hook (Rocky's chord synthesizer). Select with
  `OPENBOT_PERSONA` (default `rocky`).

See [Make your own persona](#make-your-own-persona). The physical action
vocabulary (gestures, `bullfight`/`fist bump` hardware behavior) stays in
`movement/` regardless of persona -- only the *spoken* reaction lines move into the bundle.

## Running the checks

Every `common/` module has a small `assert`-based self-check:

```bash
for m in bounded state health motor_client policy cognition mic_stream stt events sensors persona reply_schema speak_client surprise vision memory journal agenda skills tools faces contacts decider jev; do
  python3 -m common.$m
done
python3 -m personas.rocky.transform
python3 -m personas.rocky.voice   # needs SDL_AUDIODRIVER=dummy off-Pi
python3 -m services.chat --check  # who WhatsApp messages are answered from (needs neonize)
python3 -m services.tasks --check # a spoken reminder is said once, the others left to theirs
python3 -m services.ears --check  # the mic's ring, loudness, names and voice directions (no mic needed)
```

With `OPENBOT_BODY=none` every service except `alive` imports anywhere the
dependencies are installed (`requirements.txt`); with `picarx` they need the
car's libraries.

End-to-end tests and benchmarks live in `tests/` (run on the robot, from `~/openbot`, after `set -a; source openbot.env`).
They use synthetic speech (espeak) through a fake mic, and an isolated state
folder -- they never write into Rocky's real memory, and never move or speak:

```bash
python3 -m tests.test_commands_turn   # voice modes through the real turn (stubbed speech/motors)
python3 -m tests.test_voice_skills    # what's said -> the skills the real LLM picks, and what reaches the body
python3 -m tests.test_chat_skills     # the same for texts
python3 -m tests.test_mind_actions    # what the mind may pick (each skill's spec), and each one carried out
python3 -m tests.test_reminders       # tasks and reminders: each reminder delivered once, by its own service
python3 -m tests.test_listen          # listening + Mac Whisper + echo removal; prints latency
python3 -m tests.test_reply_stream    # streamed reply + photo against the live LLM
python3 -m tests.test_wake            # "Rocky" alone vs "Rocky, <instruction>", awake and asleep
python3 -m tests.test_speculation     # reply started during the pause: latency, mid-sentence pauses
python3 -m tests.bench_whisper        # Pi tiny/base vs Mac large-v3-turbo
python3 -m tests.bench_vision         # photo cost on LLM latency; face detect/recognize per frame
sudo SDL_AUDIODRIVER=dummy python3 -m tests.bench_tts   # time to Rocky's first sound
```

## Deploy & run

```bash
./deploy.sh user@robot                              # only if you edit on another machine
sudo systemd/install.sh                             # (on the robot) install/refresh + restart all units
systemctl status 'openbot-*'
journalctl -u openbot-wake-listen -f                # tail one service's logs
```

`install.sh` copies `openbot.env` to `/etc/openbot/openbot.env` and fills the
units in for your user and folder -- re-run it after changing settings.

## Known gaps (not built in this pass)

- **`vision/tracking.py` was not ported.** Face tracking was already
  disabled by default (a documented live `vilib` deadlock). The mind sees
  via one-shot captures instead (see "Feeling alive" above); continuous
  face *tracking* (turning to follow you) is still a natural follow-up.
- **sherpa-onnx/SenseVoice** are a clean third STT tier on the same
  `common/stt.py` interface, not wired up -- two backends already give a
  real fallback.
- **LED status feedback wasn't ported.** The original blinked an LED
  during listen/think/say as a visual cue; a nice-to-have, not safety- or
  architecture-relevant, dropped from this pass rather than guessed at.
- **English only**: the Vosk wake-word models and Whisper prompts are English.
- **No-body mode is new** -- verified off the robot and with the live LLM tests on the Pi;
  only tested with MLX Serve so far (Ollama is the default but untested).

## License

MIT (see `LICENSE`). Rocky's chord voice is ported from
[lahirumaramba/rocky](https://github.com/lahirumaramba/rocky) (MIT). OpenBot
runs, but doesn't include, [Basic Memory](https://github.com/basicmachines-co/basic-memory)
(AGPL-3.0, unmodified, as its own service) and, on a PiCar-X, SunFounder's
`picarx` / `robot_hat` libraries (GPL), installed separately by `setup.sh --picarx`.
