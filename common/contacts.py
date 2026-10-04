"""Who's texting: WhatsApp numbers linked to people, learned like faces.
"My name is Atul" (or, for someone it already knows, "this is atul") texted
from an allowed number links that number to Atul -- the same person Rocky
knows by face (state/faces/atul.npy) and in its notes (people/atul.md, which
gets "Atul texts me on WhatsApp"). The latest introduction wins: one phone,
one person. The numbers themselves stay next to the WhatsApp login, in
~/.openbot-whatsapp/ (0700) -- the dashboard serves everything under state/.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import faces, memory

PATH = Path.home() / ".openbot-whatsapp" / "people.json"
# Any of these + a name Rocky already knows, any case ("this atul texting you"). Not "Anna's phone".
KNOWN_INTRO = re.compile(r"\b(?i:my name is|call me|i am|i'm|im|this is|it's|its|this)\s+([A-Za-z]+)\b(?!['’]s)")


def load() -> dict[str, str]:
    """{number (digits, country code first): name}."""
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def name_of(number: str) -> str | None:
    return load().get(number)


def names(numbers) -> list[str]:
    """The linked names among `numbers` (the allowed ones), sorted."""
    people = load()
    return sorted({people[n] for n in numbers if n in people})


def introduced(text: str, strict: bool = False) -> str | None:
    """The name someone texting just gave, or None. One Rocky knows (a face or
    a note) counts however it's typed; a new one only the way a face is
    learned -- "My name is Atul", capitalized -- so "i am tired" isn't a name.
    strict: only that real introduction counts."""
    if not strict:
        known = {p.stem for p in faces.KNOWN_DIR.glob("*.npy")} | \
                {p.stem for p in (memory.MIND_DIR / "people").glob("*.md")}
        for m in KNOWN_INTRO.finditer(text):
            if memory.slug(m.group(1)) in known:
                return m.group(1).capitalize()
    name = faces.introduced_name(text)
    return name if name and not re.search(rf"\b{name}['’]s\b", text) else None  # "I am Anna's dad"


def learn(number: str, text: str, photo: bool = False) -> str | None:
    """Links `number` from what they texted; the name if that's news. Never from
    a photo's caption ("this is anna" is about the photo), and once linked, only
    a real introduction ("My name is Anna") changes it -- a loose "this is anna"
    is then about someone else. The same caution as faces: a wrong guess sticks."""
    if photo:
        return None
    name = introduced(text, strict=name_of(number) is not None)
    return name if name and link(number, name) else None


def link(number: str, name: str) -> bool:
    """`number` is `name`'s now; True if that's news."""
    people = load()
    if people.get(number) == name:
        return False
    people[number] = name
    PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(people, indent=2), encoding="utf-8")
    os.replace(tmp, PATH)
    return True


def demo() -> None:
    import shutil, tempfile
    global PATH
    orig, test_dir = (PATH, faces.KNOWN_DIR, memory.MIND_DIR), Path(tempfile.mkdtemp())
    PATH, faces.KNOWN_DIR, memory.MIND_DIR = test_dir / "people.json", test_dir / "faces", test_dir / "mind"
    try:
        (test_dir / "faces").mkdir()
        (test_dir / "faces" / "anna.npy").touch()  # a face Rocky knows
        assert introduced("My name is Atul") == "Atul" and introduced("I'm Priya") == "Priya"
        assert introduced("this anna texting you") == "Anna" and introduced("its anna") == "Anna"  # known: any case
        assert introduced("this atul texting you") is None  # not known yet: must be "My name is Atul"
        assert introduced("i am tired") is None and introduced("This is me") is None
        assert introduced("I am Anna's dad") is None and introduced("this is anna's phone") is None
        assert name_of("919876543210") is None
        assert link("919876543210", "Atul") and not link("919876543210", "Atul")  # news once
        assert link("919876543210", "Anna") and name_of("919876543210") == "Anna"  # the latest wins
        assert names(["919876543210", "15550100"]) == ["Anna"]
        assert learn("1", "this anna texting you") == "Anna"  # not linked yet: the loose form, for someone known
        assert learn("1", "this is anna") is None  # already hers: no news
        link("2", "Atul")
        assert learn("2", "this is anna") is None and name_of("2") == "Atul"  # linked: a mention never relinks
        assert learn("3", "this is anna", photo=True) is None and name_of("3") is None  # about the photo
        assert learn("2", "My name is Anna") == "Anna"  # a real introduction does
    finally:
        PATH, faces.KNOWN_DIR, memory.MIND_DIR = orig
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    demo()
    print("contacts: ok")
