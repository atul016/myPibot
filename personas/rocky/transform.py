"""Rocky's speech-pattern transform: deterministic style pass on LLM output
-- broken grammar, no articles, repeated emphasis words -- so the model's
own English doesn't have to reliably produce this in-character (confirmed
unreliable in practice). Persona-specific text processing, not framework
logic; entirely generic Python otherwise.
"""
import re

ARTICLES = {'a', 'an', 'the'}
AUXILIARIES = {'is', 'are', 'was', 'were', 'will', 'would', 'should', 'could',
               'do', 'does', 'did', 'has', 'have', 'had', 'am', 'been', 'being'}

CONTRACTIONS = {
    "your": "you", "i'm": "I", "i've": "I", "i'll": "I", "i'd": "I",
    "you're": "you", "you've": "you", "you'll": "you",
    "we're": "we", "we've": "we", "we'll": "we",
    "they're": "they", "they've": "they", "they'll": "they",
    "he's": "he", "she's": "she", "it's": "it", "that's": "that", "there's": "there",
    "what's": "what",
    "don't": "no", "doesn't": "no", "didn't": "no",
    "can't": "no can", "cannot": "no can", "won't": "no will",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "no have", "hasn't": "no have", "hadn't": "no have",
}

EMPHASIS_MAP = {
    'amazing': 'amaze amaze amaze', 'wonderful': 'amaze amaze amaze',
    'incredible': 'amaze amaze amaze', 'fantastic': 'amaze amaze amaze',
    'excellent': 'good good good', 'great': 'good good good',
    'terrible': 'bad bad bad', 'awful': 'bad bad bad', 'horrible': 'bad bad bad',
    'happy': 'happy happy happy', 'excited': 'happy happy happy',
    'sad': 'sad sad sad', 'upset': 'sad sad sad',
    'angry': 'angry angry angry', 'furious': 'angry angry angry',
    'confused': 'confuse confuse confuse',
    'scared': 'scared scared scared', 'afraid': 'scared scared scared',
    'fear': 'scared scared scared', 'worried': 'scared scared scared', 'worry': 'scared scared scared',
    'dangerous': 'danger danger danger',
    'important': 'important', 'interesting': 'interesting', 'understand': 'understand',
    'absolutely': 'yes yes yes', 'definitely': 'yes yes yes', 'certainly': 'yes yes yes',
    'impossible': 'no can. No no no', 'unfortunately': 'sad.',
}

PHRASE_MAP = [
    (r"thank you", "thank"),
    (r"radiation", "bad space ray"),
    (r"(words of )?(encouragement|encourage|motivation|motivate)", "words of encouragement"),
    (r"fist bump", "we bump fist"),
    (r"i don'?t understand", "no understand"),
    (r"i do not understand", "no understand"),
    (r"i don'?t know", "I not know"),
    (r"what do you mean", "what mean"),
    (r"what does that mean", "what mean"),
    (r"what does .+ mean", "what mean"),
    (r"i need a word for", "need word."),
    (r"i'?m going to", "I"),
    (r"going to ", ""),
    (r"want to ", "want "),
    (r"need to ", "need "),
    (r"have to ", "must "),
    (r"try to ", "try "),
    (r"\bable to ", "can "),  # was un-anchored: "unable to think" -> "uncan think" (confirmed live)
    (r"in order to ", "to "),
    (r"because of ", "because "),
    (r"a lot of ", "many "),
    (r"lots of ", "many "),
    (r"kind of ", ""),
    (r"sort of ", ""),
    (r"right now", "now"),
    (r"at this point", "now"),
    (r"at the moment", "now"),
    (r"as well", "also"),
    (r"in addition", "also"),
    (r"however", "but"),
    (r"therefore", "so"),
    (r"nevertheless", "but"),
    (r"furthermore", "also"),
    (r"approximately", "about"),
    (r"regarding", "about"),
    (r"concerning", "about"),
    (r"it seems like", "maybe"),
    (r"it appears that", "maybe"),
    (r"i think that", "I think"),
    (r"i believe that", "I think"),
    (r"you know what", ""),
    (r"to be honest", ""),
    (r"basically", ""),
    (r"actually", ""),
    (r"literally", ""),
    (r"really", "very"),
    (r"extremely", "very very"),
    (r"incredibly", "very very"),
    (r"goodbye", "see you later. But I no see you later"),
]

_PHRASE_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in PHRASE_MAP]


def rocky_transform(text: str) -> str:
    """Transform English text into Rocky's speech patterns."""
    if not text or not text.strip():
        return text

    text = re.sub(r'\[\s*chord\s*:[^\]]*\]', '', text, flags=re.IGNORECASE)

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue

        is_question = s.endswith('?')

        for pattern, replacement in _PHRASE_PATTERNS:
            s = pattern.sub(replacement, s)

        words = s.split()
        new_words = []
        for w in words:
            lower = w.lower().rstrip('.,!?;:')
            punct = w[len(lower):] if len(w) > len(lower) else ''

            if lower in CONTRACTIONS:
                new_words.append(CONTRACTIONS[lower] + punct)
            elif lower in EMPHASIS_MAP:
                new_words.append(EMPHASIS_MAP[lower] + punct)
            elif lower in ARTICLES:
                continue
            elif lower in AUXILIARIES and len(new_words) > 0:
                continue
            else:
                new_words.append(w)

        s = ' '.join(new_words)
        s = re.sub(r'\s+', ' ', s).strip()

        if is_question and 'question' not in s.lower():
            s = s.rstrip('?').strip() + ', question?'
        elif is_question:
            s = s.rstrip('?').strip() + '?'

        if s:
            s = s[0].upper() + s[1:]

        result.append(s)

    output = ' '.join(result)
    output = re.sub(r'\s+', ' ', output)
    output = re.sub(r'\s+([.,!?])', r'\1', output)
    output = re.sub(r'\.\.+', '.', output)
    output = re.sub(r',\s*\.', '.', output)

    return output.strip()


def demo() -> None:
    assert rocky_transform("") == ""
    assert "amaze amaze amaze" in rocky_transform("That is amazing!")
    assert rocky_transform("What is your name?").endswith("question?")
    assert "we bump fist" in rocky_transform("Let's fist bump")


if __name__ == "__main__":
    demo()
    print("rocky/transform: ok")
