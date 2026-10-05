"""Skills: everything the robot can do, a folder each in skills/ -- the LLM
reads them and decides what to do, instead of phrase lists matched in code or
rules written into prompts. A skill is two halves:

  skills/<name>/<name>.md   the knowing: what it does, when (and when not) to use
                            it, what it needs to know -- its one spec: the decision's
                            schema, the mind's list, Jev's options and the checking all
                            come from it. Edit it to change how it's used.
  skills/<name>/<name>.py   the muscle: run(ctx, **params) does it, and says what
                            happened in a sentence ("You won't text them first
                            until 15:20.") -- or None if there was nothing to do.

    ---
    name: pause_texting
    description: Stop texting them first for a while, because they asked you to.
    where: text, voice          the channels it can be used from (text, voice, mind)
    needs: body                 optional: only with a body that drives (config.CAN_DRIVE)
    effect: audio               optional, for the mind: audio or motion (throttled and gated), presence (gated)
    tool: yes                   optional, for the mind: it sees what run() returns, then decides again
    mind: Only when ...         optional: what the mind's list says instead of the text below
    params:
      minutes (integer, 1-720): how long -- 60 is an hour
      change (louder|softer): one of these
    ---
    When to use it, when not, examples.

A param is `name (type, ...): what it is`. Types: string, integer, number,
boolean, a|b|c (one of these), or one of LISTS (this body's own -- a skill with
an empty one isn't offered). Then, optionally: `max N` (characters, at least
one), `N-M` (a range), `default X` (it may be left out).

A turn is three steps, like a person: decide (an LLM call that sees every skill
people may ask the body for -- menu() -- and answers only with decision_model()'s
`actions`), do (read_decision() keeps what's usable from this channel and names
the rest as refused; run() each), then say (the persona's own call, told what was
done -- so its words can't claim what didn't happen). Reflexes stay in code (an
instant "stop", edge and battery safety): a hand off a hot stove doesn't wait
to think. The mind (services/mind.py) picks among its own (`where: mind`) from
mind_menu(), and check() holds its pick to the same spec.
"""
from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BeforeValidator, Field, TypeAdapter, ValidationError, create_model

import config as cfg

from . import agenda, memory

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
PEOPLE = frozenset({"text", "voice"})  # the channels people ask from -- the mind's own skills aren't theirs
_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool}
LISTS = {  # what a param may be, on this body
    "direction": lambda: cfg.LOOK_DIRECTIONS,
    "gesture": lambda: cfg.TONE_ACTIONS,
    "sound": lambda: cfg.SOUNDS,
    "memory_kind": lambda: sorted(memory.KINDS),
    "watch_kind": lambda: sorted(agenda.WATCHABLE - (set() if cfg.HAS_BODY else agenda.NEEDS_BODY)),
}


@dataclass
class Skill:
    name: str
    description: str
    where: frozenset[str]
    needs: frozenset[str]
    params: dict[str, tuple[str, str]]  # name -> (type, description)
    body: str
    tool: bool = False
    effect: str = ""
    mind: str = ""  # the mind's own guidance, for a skill people use too

    def line(self) -> str:
        args = ", ".join(self.params)
        return f"- {self.name}({args}): {self.description} {' '.join(self.body.split())}"


def parse(text: str, stem: str) -> Skill:
    m = re.match(r"---\n(.*?)\n---\n?(.*)", text, re.S)
    if not m:
        raise ValueError(f"{stem}.md: no --- header ---")
    meta: dict[str, str] = {}
    params: dict[str, tuple[str, str]] = {}
    key = ""
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        if line[0] in " \t":
            p = re.match(r"\s+(\w+)\s*\(([^)]+)\)\s*:\s*(.+)", line)
            if key != "params" or not p:
                raise ValueError(f"{stem}.md: can't read {line.strip()!r} -- params are `  name (type): what it is`")
            try:
                _spec(p.group(2))
            except ValueError as e:
                raise ValueError(f"{stem}.md: {p.group(1)}: {e}") from None
            params[p.group(1)] = (p.group(2).strip(), p.group(3).strip())
        else:
            key, _, value = (s.strip() for s in line.partition(":"))
            meta[key] = value
    name = meta.get("name", stem)
    if name != stem or not meta.get("description"):
        raise ValueError(f"{stem}.md: needs `name: {stem}` and a description")
    if meta.get("effect", "") not in ("", "audio", "motion", "presence") or meta.get("tool", "no") not in ("yes", "no"):
        raise ValueError(f"{stem}.md: effect is audio, motion or presence; tool is yes or no")
    words = lambda v: frozenset(w.strip() for w in v.split(",") if w.strip())  # noqa: E731
    return Skill(name, meta["description"], words(meta.get("where", "")), words(meta.get("needs", "")), params,
                 m.group(2).strip(), meta.get("tool") == "yes", meta.get("effect", ""), meta.get("mind", ""))


def _spec(spec: str) -> tuple[str, dict[str, Any]]:
    """`integer, 1-720` -> ("integer", {"ge": 1, "le": 720}); `max N`, `default X` likewise."""
    typ, *mods = (m.strip() for m in spec.split(","))
    if typ not in _TYPES and typ not in LISTS and "|" not in typ:
        raise ValueError(f"unknown type {typ!r} -- {', '.join(_TYPES)}, a|b|c or one of {', '.join(LISTS)}")
    kw: dict[str, Any] = {}
    for m in mods:
        if r := re.fullmatch(r"(\d+)-(\d+)", m):
            kw.update(ge=int(r[1]), le=int(r[2]))
        elif r := re.fullmatch(r"max (\d+)", m):
            kw.update(min_length=1, max_length=int(r[1]))
        elif r := re.fullmatch(r"default (.+)", m):
            kw["default"] = {"integer": int, "number": float, "boolean": lambda v: v == "true"}.get(typ, str)(r[1])
        else:
            raise ValueError(f"can't read {m!r} -- `max N`, `N-M` or `default X`")
    return typ, kw


def _choices(typ: str) -> list[str] | None:
    """The values a param of this type may take -- None: any of its type."""
    if "|" in typ:
        return [c.strip() for c in typ.split("|")]
    return list(LISTS[typ]()) if typ in LISTS else None


def _one_line(v: Any) -> Any:
    return " ".join(v.split()) if isinstance(v, str) else v


def _model(s: Skill) -> Any:
    """{"skill": name, **params}, each param held to its spec."""
    fields: dict[str, Any] = {"skill": (Literal[s.name], ...)}
    for p, (spec, what) in s.params.items():
        typ, kw = _spec(spec)
        choices = _choices(typ)
        kind = Literal[tuple(choices)] if choices is not None else \
            Annotated[str, BeforeValidator(_one_line)] if typ == "string" else _TYPES[typ]
        fields[p] = (kind, Field(description=what, **kw))
    return create_model(f"Use_{s.name}", **fields)


def _offered(s: Skill) -> bool:
    return all(_choices(_spec(spec)[0]) != [] for spec, _ in s.params.values())


def load(directory: Path | None = None) -> dict[str, Skill]:
    """Every skills/<name>/<name>.md -- read fresh each turn, so an edited file counts at once."""
    directory = directory or SKILLS_DIR
    return {p.stem: parse(p.read_text(encoding="utf-8"), p.stem) for p in sorted(directory.glob("*/*.md"))
            if p.stem == p.parent.name}


def available(channel: str | None, has_body: bool, skills: dict[str, Skill]) -> list[Skill]:
    """The skills usable from `channel` -- None: every one people may ask this body for (by text or voice)."""
    return [s for s in skills.values() if (channel in s.where if channel else s.where & PEOPLE)
            and (has_body or "body" not in s.needs) and _offered(s)]


def menu(has_body: bool, skills: dict[str, Skill]) -> str:
    """Every skill people may ask this body for, for the decision -- whether this channel
    may use it is the code's call afterwards, so the LLM can name what was really asked."""
    return "\n".join(s.line() for s in available(None, has_body, skills))


def param_words(spec: str, what: str) -> str:
    """A param as the mind's list shows it: what it is, and what it may be."""
    typ, kw = _spec(spec)
    choices = _choices(typ)
    if choices is not None:
        return f"{what} -- one of: {cfg.describe_actions(choices)}"
    return what + (f" ({kw['ge']}-{kw['le']})" if "le" in kw else
                   f" (at most {kw['max_length']} chars)" if "max_length" in kw else "")


def mind_menu(skills: list[Skill]) -> str:
    """The mind's actions, `  name {"param": what it may be}  -- what it's for`; its tools first."""
    lines = []
    for s in sorted(skills, key=lambda s: not s.tool):
        params = ", ".join(f"{json.dumps(p)}: {param_words(spec, what)}" for p, (spec, what) in s.params.items())
        lines.append(f"  {s.name} {{{params}}}  -- {'TOOL: ' if s.tool else ''}"
                     f"{s.description} {s.mind or ' '.join(s.body.split())}".rstrip())
    return "\n".join(lines)


def action_model(channel: str | None, has_body: bool, skills: dict[str, Skill]) -> Any:
    """The type of one action: any skill usable from `channel` (None: any people may ask for), with its params."""
    models = [_model(s) for s in available(channel, has_body, skills)]
    if not models:
        return None
    return models[0] if len(models) == 1 else Union[tuple(models)]


def decision_model(has_body: bool, skills: dict[str, Skill]) -> Any:
    """{"actions": [...]} -- the decision call's whole answer."""
    action = action_model(None, has_body, skills)
    if action is None:
        return None
    return create_model("Decision", actions=(list[action], Field(
        default_factory=list, description="Each skill their message asks for -- empty when it asks for none.")))


def read_decision(raw: str, channel: str, has_body: bool, skills: dict[str, Skill]) -> tuple[list, list[str]]:
    """(the actions to carry out -- valid, and usable from `channel`; the skills it
    asked for that aren't usable from there). Leniently: the LLM server doesn't
    always hold the JSON to its schema."""
    action = action_model(None, has_body, skills)
    try:
        items = json.loads(raw).get("actions") or []
    except (ValueError, AttributeError):
        return [], []
    if action is None or not isinstance(items, list):
        return [], []
    check, here = TypeAdapter(action), {s.name for s in available(channel, has_body, skills)}
    known = {s.name for s in available(None, has_body, skills)}  # a mind-only skill is as unknown as "fly"
    actions, refused = [], []
    for item in items:
        name = item.get("skill") if isinstance(item, dict) else None
        if name in known and name not in here:
            refused.append(name)  # asked for, with good params or not: it can't be done from here anyway
            continue
        try:
            actions.append(check.validate_python(item))
        except ValidationError:
            pass  # a skill it can use, garbled: as if not asked
    return actions, list(dict.fromkeys(refused))


DECIDE_TEMPERATURE = 0.1  # the same words should get the same decision; replies keep their own temperature
DECIDE_RULES = ("Here you only decide what to DO about it: put each skill their words ask for in `actions` -- only "
                "what they ask, not what you'd like to do; none for plain chat, questions or news. Your words come "
                "after, separately. If their words don't clearly ask for something, choose nothing -- they can "
                "always ask again. When they ask, do it -- it's their call.\n\nYour skills:\n")


def last_turn(history: list[dict]) -> str:
    """The exchange just before, told as a fact. Given as chat, the decision carried the talk on:
    "Rocky." after "turn towards me" turned left again, "Computer." drove forward (2026-10-04)."""
    for i in range(len(history) - 1, 0, -1):
        if history[i]["role"] == "assistant" and history[i - 1]["role"] == "user":
            reply = history[i]["content"]
            try:
                reply = json.loads(reply).get("reply", reply)
            except (ValueError, AttributeError):
                pass
            return (f'A moment ago they said "{history[i - 1]["content"]}" and you answered "{reply}" -- that is '
                    'done; nothing more to do about it unless their words now ask ("yes" to something you offered, '
                    '"again").\n')
    return ""


def decide(base_url: str, model: str, channel: str, has_body: bool, intro: str, situation: str, text: str,
           history: list[dict], skills: dict[str, Skill] | None = None) -> tuple[list, list[str]]:
    """Step 1 of a turn: which skills `text` asks for -- a short LLM call that
    sees every skill the body has and answers only with them; no persona, no
    memories (old "Moving forward!" lines taught it to claim moves), and of the
    conversation only last_turn(). `intro`: how the words came ("Someone just
    texted you on WhatsApp."). Returns read_decision()'s (actions usable from
    `channel`, skills refused)."""
    from . import cognition  # deferred: the parsing above runs without the network stack
    skills = skills or load()
    Decision = decision_model(has_body, skills)
    if Decision is None:
        return [], []
    result = cognition.ask(base_url, model, f"{situation}\n{last_turn(history)}Their words: {text}",
                           system=f"You're a small robot. {intro} {DECIDE_RULES}{menu(has_body, skills)}",
                           json_schema=Decision.model_json_schema(),
                           temperature=DECIDE_TEMPERATURE, num_predict=200)
    if result.status != cognition.AVAILABLE:
        print(f"skills: deciding: cognition {result.status}: {result.error}")
        return [], []
    return read_decision(result.text, channel, has_body, skills)


def check(skill: Skill, params: dict) -> Any:
    """`params` held to `skill`'s spec: the action run() takes. A ValidationError (a ValueError) if they don't fit."""
    return _model(skill).model_validate({**params, "skill": skill.name})


def use(name: str, ctx: dict, params: dict, skills: dict[str, Skill] | None = None) -> str | None:
    """attempt() one skill by name, its params check()ed -- for code that already knows which
    (the mind's own pick; a texted skill handed to openbot-wake-listen to do at home)."""
    skills = skills or load()
    try:
        action = check(skills[name], params)
    except (KeyError, ValueError) as e:
        print(f"skills: can't use {name} {params}: {e}")
        return None
    return attempt(action, ctx)


def attempt(action: Any, ctx: dict) -> str | None:
    """run(), but a failing skill fails only itself -- never the turn it's part of."""
    try:
        return run(action, ctx)
    except Exception as e:
        print(f"skills: {getattr(action, 'skill', '?')} failed: {e!r}")
        return None


def refusal(skill: Skill, channel: str) -> str:
    """What a reply is told when they asked, from `channel`, for a skill that can't be used from there."""
    here = {"text": "by text", "voice": "from a conversation out loud"}.get(channel, "from here")
    there = ("when someone asks you out loud, in person" if "voice" in skill.where else
             "when someone texts you" if "text" in skill.where else "on your own")
    what = skill.description[0].lower() + skill.description[1:].rstrip(".")
    return (f"They asked you to {what}. You can't do that {here} -- only {there}. Tell them so, and don't say "
            "you're doing it.")


def run(action: Any, ctx: dict, directory: Path | None = None) -> str | None:
    """Carry out one chosen action: skills/<name>/<name>.py's run(ctx, **params).
    Returns what happened, in a sentence for the reply -- or None."""
    params = action.model_dump()
    name = params.pop("skill")
    path = (directory or SKILLS_DIR) / name / f"{name}.py"
    if not path.exists():
        print(f"skills: {name} has no muscle ({path}) -- nothing done")
        return None
    spec = importlib.util.spec_from_file_location(f"skills.{name}", path)
    muscle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(muscle)
    return muscle.run(ctx, **params)


def demo() -> None:
    import tempfile
    d = Path(tempfile.mkdtemp())
    for name in ("nap", "drive", "think", "note", "honk"):
        (d / name).mkdir()
    (d / "nap" / "nap.md").write_text("---\nname: nap\ndescription: Take a nap.\nwhere: text, voice\nparams:\n"
                                      "  minutes (integer, 1-720): how long\n  how (light|deep, default light): what kind"
                                      "\n  dream (string, max 10, default sheep): of what\n---\nWhen tired.\n")
    (d / "drive" / "drive.md").write_text("---\nname: drive\ndescription: Drive somewhere.\nwhere: voice\n"
                                          "needs: body\n---\n")
    (d / "think" / "think.md").write_text("---\nname: think\ndescription: Think.\nwhere: mind\ntool: yes\n---\n")
    (d / "note" / "note.md").write_text("---\nname: note\ndescription: Note it.\nwhere: mind, text\neffect: audio\n"
                                        "mind: When it matters.\nparams:\n  kind (memory_kind): what about\n---\n"
                                        "When they ask.\n")
    (d / "honk" / "honk.md").write_text("---\nname: honk\ndescription: Honk.\nwhere: mind\nparams:\n"
                                        "  name (nothing): which\n---\n")
    LISTS["nothing"] = lambda: []  # a body without any: not offered
    skills = load(d)
    assert skills["nap"].params["minutes"] == ("integer, 1-720", "how long")
    assert skills["drive"].needs == {"body"} and skills["nap"].where == {"text", "voice"}
    assert skills["think"].tool and skills["note"].effect == "audio" and not skills["nap"].tool
    assert [s.name for s in available("text", True, skills)] == ["nap", "note"]
    assert [s.name for s in available(None, False, skills)] == ["nap", "note"]  # no body, no driving; not the mind's
    assert [s.name for s in available("mind", True, skills)] == ["note", "think"]  # honk: no sounds on this body
    assert "- nap(minutes, how, dream): Take a nap. When tired." in menu(True, skills) and "drive(" in menu(True, skills)
    assert "drive(" not in menu(False, skills) and "think(" not in menu(True, skills)
    mind = mind_menu(available("mind", True, skills))
    assert mind.startswith("  think {}  -- TOOL") and '"kind": what about -- one of: lesson; person; place; self' in mind
    assert "Note it. When it matters." in mind and "When they ask." not in mind  # the mind's own line, not people's
    assert "- note(kind): Note it. When they ask." in menu(True, skills)
    # one spec, checked: a number from text, spaces squeezed, defaults, and what's out of range or too long
    nap = check(skills["nap"], {"minutes": "5", "dream": "  big   fish "}).model_dump()
    assert nap == {"skill": "nap", "minutes": 5, "how": "light", "dream": "big fish"}
    for bad in ({"minutes": 0}, {"minutes": 721}, {"minutes": "soon"}, {"minutes": 5, "how": "long"},
                {"minutes": 5, "dream": "x" * 11}, {"minutes": 5, "dream": "   "}):
        try:
            check(skills["nap"], bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    schema = decision_model(True, skills).model_json_schema()
    assert list(schema["properties"]) == ["actions"] and "Use_think" not in schema["$defs"]
    assert schema["$defs"]["Use_nap"]["properties"]["minutes"]["maximum"] == 720  # what it's shown is what's checked
    # by text: nap is done, drive is refused by name (even garbled), junk and the mind's own are dropped
    acts, refused = read_decision('{"actions": [{"skill": "nap", "minutes": 20, "how": "deep"}, {"skill": "drive", '
                                  '"where": "door"}, {"skill": "nap", "minutes": "x"}, {"skill": "fly"}, '
                                  '{"skill": "think"}, 7]}', "text", True, skills)
    assert [a.skill for a in acts] == ["nap"] and acts[0].minutes == 20 and refused == ["drive"]
    assert read_decision("not json", "text", True, skills) == ([], [])
    assert read_decision('{"actions": "nap"}', "text", True, skills) == ([], [])
    (d / "nap" / "nap.py").write_text("def run(ctx, minutes, how, dream):\n    ctx['slept'] = minutes\n"
                                      "    return f'You took a {how} nap.'\n")
    ctx: dict = {}
    assert run(acts[0], ctx, d) == "You took a deep nap." and ctx["slept"] == 20  # the muscle, from its folder
    assert use("nap", ctx, {"minutes": 9}, skills) is None  # its muscle isn't where load() looks: nothing done
    assert use("nap", ctx, {"minutes": 0}, skills) is None and use("fly", ctx, {}, skills) is None
    for broken in ("no header", "---\nname: other\ndescription: x\n---\n", "---\nname: nap\n---\n",
                   "---\nname: nap\ndescription: x\nparams:\n  minutes: how long\n---\n",
                   "---\nname: nap\ndescription: x\nparams:\n  minutes (integer, sometimes): how long\n---\n",
                   "---\nname: nap\ndescription: x\nparams:\n  colour (colour): which\n---\n",
                   "---\nname: nap\ndescription: x\neffect: loud\n---\n"):
        try:
            parse(broken, "nap")
            raise AssertionError(broken)
        except ValueError:
            pass
    del LISTS["nothing"]
    talk = [{"role": "user", "content": "Turn left."}, {"role": "assistant", "content": '{"reply": "Turning!"}'},
            {"role": "user", "content": "(nobody said anything yet)"}]
    assert last_turn(talk).startswith('A moment ago they said "Turn left." and you answered "Turning!"')
    assert last_turn([]) == "" and "you answered \"Hi\"" in last_turn(talk[:1] + [{"role": "assistant", "content": "Hi"}])
    for s in load().values():  # every shipped skill file reads, and has its muscle
        assert (SKILLS_DIR / s.name / f"{s.name}.py").exists(), f"skills/{s.name}/{s.name}.py missing"
        _model(s) if _offered(s) else None  # and its params make a model


if __name__ == "__main__":
    demo()
    print("skills: ok")
