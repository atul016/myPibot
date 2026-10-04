"""Skills: everything the robot can do, a folder each in skills/ -- the LLM
reads them and decides what to do, instead of phrase lists matched in code or
rules written into prompts. A skill is two halves:

  skills/<name>/<name>.md   the knowing: what it does, when (and when not) to use
                            it, what it needs to know. Edit it to change how it's used.
  skills/<name>/<name>.py   the muscle: run(ctx, **params) does it, and says what
                            happened in a sentence ("You won't text them first
                            until 15:20.") -- or None if there was nothing to do.

    ---
    name: pause_texting
    description: Stop texting them first for a while, because they asked you to.
    where: text, voice          the channels it can be used from (text, voice, mind)
    needs: body                 optional: only with a body that drives (config.CAN_DRIVE)
    params:
      minutes (integer): how long -- 60 is an hour
      change (louder|softer): one of these
    ---
    When to use it, when not, examples.

A turn is three steps, like a person: decide (an LLM call that sees every skill
the body has -- menu() -- and answers only with decision_model()'s `actions`),
do (read_decision() keeps what's usable from this channel and names the rest
as refused; run() each), then say (the persona's own call, told what was done --
so its words can't claim what didn't happen). Reflexes stay in code (an
instant "stop", edge and battery safety): a hand off a hot stove doesn't wait
to think.
"""
from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import Field, TypeAdapter, ValidationError, create_model

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool}


@dataclass
class Skill:
    name: str
    description: str
    where: frozenset[str]
    needs: frozenset[str]
    params: dict[str, tuple[str, str]]  # name -> (type, description)
    body: str

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
            params[p.group(1)] = (p.group(2).strip(), p.group(3).strip())
        else:
            key, _, value = (s.strip() for s in line.partition(":"))
            meta[key] = value
    name = meta.get("name", stem)
    if name != stem or not meta.get("description"):
        raise ValueError(f"{stem}.md: needs `name: {stem}` and a description")
    words = lambda v: frozenset(w.strip() for w in v.split(",") if w.strip())  # noqa: E731
    return Skill(name, meta["description"], words(meta.get("where", "")), words(meta.get("needs", "")), params,
                 m.group(2).strip())


def load(directory: Path | None = None) -> dict[str, Skill]:
    """Every skills/<name>/<name>.md -- read fresh each turn, so an edited file counts at once."""
    directory = directory or SKILLS_DIR
    return {p.stem: parse(p.read_text(encoding="utf-8"), p.stem) for p in sorted(directory.glob("*/*.md"))
            if p.stem == p.parent.name}


def available(channel: str | None, has_body: bool, skills: dict[str, Skill]) -> list[Skill]:
    """The skills usable from `channel` -- None: every skill this body has."""
    return [s for s in skills.values() if (channel is None or channel in s.where)
            and (has_body or "body" not in s.needs)]


def menu(has_body: bool, skills: dict[str, Skill]) -> str:
    """Every skill this body has, for the decision -- whether this channel may use
    it is the code's call afterwards, so the LLM can name what was really asked."""
    return "\n".join(s.line() for s in available(None, has_body, skills))


def _param_type(typ: str) -> Any:
    if "|" in typ:
        return Literal[tuple(t.strip() for t in typ.split("|"))]
    return _TYPES[typ]


def action_model(channel: str | None, has_body: bool, skills: dict[str, Skill]) -> Any:
    """The type of one action: any skill usable from `channel` (None: any), with its params."""
    models = []
    for s in available(channel, has_body, skills):
        fields: dict[str, Any] = {"skill": (Literal[s.name], ...)}
        for p, (typ, what) in s.params.items():
            fields[p] = (_param_type(typ), Field(description=what))
        models.append(create_model(f"Use_{s.name}", **fields))
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
    actions, refused = [], []
    for item in items:
        name = item.get("skill") if isinstance(item, dict) else None
        if name in skills and name not in here:
            refused.append(name)  # asked for, with good params or not: it can't be done from here anyway
            continue
        try:
            actions.append(check.validate_python(item))
        except ValidationError:
            pass  # a skill it can use, garbled: as if not asked
    return actions, list(dict.fromkeys(refused))


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
    for name in ("nap", "drive", "think"):
        (d / name).mkdir()
    (d / "nap" / "nap.md").write_text("---\nname: nap\ndescription: Take a nap.\nwhere: text, voice\nparams:\n"
                                      "  minutes (integer): how long\n  how (light|deep): what kind\n---\nWhen tired.\n")
    (d / "drive" / "drive.md").write_text("---\nname: drive\ndescription: Drive somewhere.\nwhere: voice\n"
                                          "needs: body\n---\n")
    (d / "think" / "think.md").write_text("---\nname: think\ndescription: Think.\nwhere: mind\n---\n")
    skills = load(d)
    assert skills["nap"].params == {"minutes": ("integer", "how long"), "how": ("light|deep", "what kind")}
    assert skills["drive"].needs == {"body"} and skills["nap"].where == {"text", "voice"}
    assert [s.name for s in available("text", True, skills)] == ["nap"]
    assert [s.name for s in available(None, False, skills)] == ["nap", "think"]  # no body, no driving
    assert "- nap(minutes, how): Take a nap. When tired." in menu(True, skills) and "drive(" in menu(True, skills)
    assert "drive(" not in menu(False, skills)
    Decision = decision_model(True, skills)
    assert list(Decision.model_json_schema()["properties"]) == ["actions"]
    # by text: nap is done, drive is refused by name (even garbled), junk is dropped
    acts, refused = read_decision('{"actions": [{"skill": "nap", "minutes": 20, "how": "light"}, {"skill": "drive", '
                                  '"where": "door"}, {"skill": "nap", "minutes": "x"}, {"skill": "fly"}, 7]}',
                                  "text", True, skills)
    assert [a.skill for a in acts] == ["nap"] and acts[0].minutes == 20 and refused == ["drive"]
    assert read_decision("not json", "text", True, skills) == ([], [])
    assert read_decision('{"actions": "nap"}', "text", True, skills) == ([], [])
    (d / "nap" / "nap.py").write_text("def run(ctx, minutes, how):\n    ctx['slept'] = minutes\n"
                                      "    return f'You took a {how} nap.'\n")
    ctx: dict = {}
    assert run(acts[0], ctx, d) == "You took a light nap." and ctx["slept"] == 20  # the muscle, from its folder
    for broken in ("no header", "---\nname: other\ndescription: x\n---\n", "---\nname: nap\n---\n",
                   "---\nname: nap\ndescription: x\nparams:\n  minutes: how long\n---\n"):
        try:
            parse(broken, "nap")
            raise AssertionError(broken)
        except ValueError:
            pass
    for s in load().values():  # every shipped skill file reads; the texting ones all have their muscle
        if "text" in s.where:
            assert (SKILLS_DIR / s.name / f"{s.name}.py").exists(), f"skills/{s.name}/{s.name}.py missing"


if __name__ == "__main__":
    demo()
    print("skills: ok")
