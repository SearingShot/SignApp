"""
Hybrid English → ASL Gloss Converter

Combines three strategies for accurate gloss generation:
  1. Rule-based grammar transforms (drop articles/copulas, reorder)
  2. NLTK WordNet lemmatizer for verb/noun normalization
  3. Comprehensive gloss lookup dictionary for idioms & common phrases
"""

import re
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only once)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)

_lemmatizer = WordNetLemmatizer()

# ── Words to drop (ASL omits these) ────────────────────────────────
ARTICLES = {"a", "an", "the"}
COPULAS = {"is", "am", "are", "was", "were", "be", "been", "being"}
AUXILIARIES = {"do", "does", "did", "will", "would", "shall", "should",
               "can", "could", "may", "might", "must", "has", "have", "had"}
PREPOSITIONS_DROP = {"to", "of"}  # commonly dropped in ASL
FILLER_WORDS = {"just", "really", "very", "so", "um", "uh", "like",
                "well", "actually", "basically", "literally"}

DROP_WORDS = ARTICLES | COPULAS | FILLER_WORDS

# ── Phrase-level gloss dictionary (multi-word → single sign) ───────
PHRASE_GLOSSARY: dict[str, str] = {
    "thank you": "THANK-YOU",
    "thanks": "THANK-YOU",
    "how are you": "HOW YOU",
    "what's up": "WHAT-UP",
    "good morning": "GOOD MORNING",
    "good night": "GOOD NIGHT",
    "good afternoon": "GOOD AFTERNOON",
    "excuse me": "EXCUSE",
    "i'm sorry": "SORRY",
    "i am sorry": "SORRY",
    "a lot": "MANY",
    "don't": "NOT",
    "doesn't": "NOT",
    "didn't": "NOT",
    "can't": "CAN NOT",
    "cannot": "CAN NOT",
    "won't": "WILL NOT",
    "wouldn't": "WILL NOT",
    "shouldn't": "SHOULD NOT",
    "couldn't": "CAN NOT",
    "isn't": "NOT",
    "aren't": "NOT",
    "wasn't": "NOT",
    "weren't": "NOT",
    "i'm": "I",
    "i am": "I",
    "you're": "YOU",
    "you are": "YOU",
    "he's": "HE",
    "she's": "SHE",
    "it's": "IT",
    "we're": "WE",
    "they're": "THEY",
    "there is": "HAVE",
    "there are": "HAVE",
    "right now": "NOW",
    "a little": "LITTLE",
    "a bit": "LITTLE",
    "of course": "OF-COURSE",
    "no problem": "NO-PROBLEM",
    "long time": "LONG-TIME",
    "how much": "HOW-MUCH",
    "how many": "HOW-MANY",
}

# ── Word-level synonym/mapping dictionary ──────────────────────────
WORD_GLOSSARY: dict[str, str] = {
    # Greetings
    "hello": "HELLO",
    "hi": "HELLO",
    "hey": "HELLO",
    "goodbye": "BYE",
    "bye": "BYE",

    # Pronouns (pass-through but normalize)
    "i": "I",
    "me": "I",
    "my": "MY",
    "mine": "MY",
    "you": "YOU",
    "your": "YOUR",
    "yours": "YOUR",
    "he": "HE",
    "him": "HE",
    "his": "HIS",
    "she": "SHE",
    "her": "SHE",
    "hers": "HER",
    "it": "IT",
    "its": "IT",
    "we": "WE",
    "us": "WE",
    "our": "OUR",
    "they": "THEY",
    "them": "THEY",
    "their": "THEIR",

    # Common verbs (map to ASL base forms)
    "want": "WANT",
    "wants": "WANT",
    "wanted": "WANT",
    "wanting": "WANT",
    "need": "NEED",
    "needs": "NEED",
    "needed": "NEED",
    "like": "LIKE",
    "likes": "LIKE",
    "liked": "LIKE",
    "love": "LOVE",
    "loves": "LOVE",
    "loved": "LOVE",
    "know": "KNOW",
    "knows": "KNOW",
    "knew": "KNOW",
    "known": "KNOW",
    "think": "THINK",
    "thinks": "THINK",
    "thought": "THINK",
    "see": "SEE",
    "sees": "SEE",
    "saw": "SEE",
    "seen": "SEE",
    "help": "HELP",
    "helps": "HELP",
    "helped": "HELP",
    "go": "GO",
    "goes": "GO",
    "going": "GO",
    "went": "GO",
    "gone": "GO",
    "come": "COME",
    "comes": "COME",
    "came": "COME",
    "eat": "EAT",
    "eats": "EAT",
    "ate": "EAT",
    "eaten": "EAT",
    "drink": "DRINK",
    "drinks": "DRINK",
    "drank": "DRINK",
    "work": "WORK",
    "works": "WORK",
    "worked": "WORK",
    "working": "WORK",
    "live": "LIVE",
    "lives": "LIVE",
    "lived": "LIVE",
    "feel": "FEEL",
    "feels": "FEEL",
    "felt": "FEEL",
    "say": "SAY",
    "says": "SAY",
    "said": "SAY",
    "tell": "TELL",
    "tells": "TELL",
    "told": "TELL",
    "ask": "ASK",
    "asks": "ASK",
    "asked": "ASK",
    "give": "GIVE",
    "gives": "GIVE",
    "gave": "GIVE",
    "given": "GIVE",
    "take": "TAKE",
    "takes": "TAKE",
    "took": "TAKE",
    "taken": "TAKE",
    "make": "MAKE",
    "makes": "MAKE",
    "made": "MAKE",
    "get": "GET",
    "gets": "GET",
    "got": "GET",
    "wait": "WAIT",
    "waits": "WAIT",
    "waited": "WAIT",
    "learn": "LEARN",
    "learns": "LEARN",
    "learned": "LEARN",
    "teach": "TEACH",
    "teaches": "TEACH",
    "taught": "TEACH",
    "understand": "UNDERSTAND",
    "understands": "UNDERSTAND",
    "understood": "UNDERSTAND",
    "finish": "FINISH",
    "finished": "FINISH",
    "start": "START",
    "started": "START",
    "stop": "STOP",
    "stopped": "STOP",
    "try": "TRY",
    "tries": "TRY",
    "tried": "TRY",
    "call": "CALL",
    "called": "CALL",
    "play": "PLAY",
    "played": "PLAY",
    "run": "RUN",
    "ran": "RUN",
    "walk": "WALK",
    "walked": "WALK",
    "sit": "SIT",
    "sat": "SIT",
    "stand": "STAND",
    "stood": "STAND",
    "open": "OPEN",
    "opened": "OPEN",
    "close": "CLOSE",
    "closed": "CLOSE",
    "buy": "BUY",
    "bought": "BUY",
    "bring": "BRING",
    "brought": "BRING",
    "read": "READ",
    "write": "WRITE",
    "wrote": "WRITE",
    "written": "WRITE",
    "speak": "SPEAK",
    "spoke": "SPEAK",
    "sign": "SIGN",
    "signed": "SIGN",
    "watch": "WATCH",
    "look": "LOOK",
    "looked": "LOOK",
    "listen": "LISTEN",

    # Question words
    "what": "WHAT",
    "where": "WHERE",
    "when": "WHEN",
    "why": "WHY",
    "how": "HOW",
    "who": "WHO",
    "which": "WHICH",

    # Common nouns
    "person": "PERSON",
    "people": "PEOPLE",
    "man": "MAN",
    "woman": "WOMAN",
    "boy": "BOY",
    "girl": "GIRL",
    "child": "CHILD",
    "children": "CHILD",
    "baby": "BABY",
    "friend": "FRIEND",
    "friends": "FRIEND",
    "family": "FAMILY",
    "mother": "MOTHER",
    "mom": "MOTHER",
    "father": "FATHER",
    "dad": "FATHER",
    "brother": "BROTHER",
    "sister": "SISTER",
    "dog": "DOG",
    "cat": "CAT",
    "house": "HOUSE",
    "home": "HOME",
    "school": "SCHOOL",
    "food": "FOOD",
    "water": "WATER",
    "car": "CAR",
    "book": "BOOK",
    "phone": "PHONE",
    "name": "NAME",
    "day": "DAY",
    "today": "TODAY",
    "tomorrow": "TOMORROW",
    "yesterday": "YESTERDAY",
    "morning": "MORNING",
    "night": "NIGHT",
    "time": "TIME",
    "world": "WORLD",
    "year": "YEAR",
    "money": "MONEY",
    "job": "WORK",
    "store": "STORE",
    "door": "DOOR",
    "place": "PLACE",
    "city": "CITY",
    "country": "COUNTRY",
    "weather": "WEATHER",

    # Adjectives
    "good": "GOOD",
    "bad": "BAD",
    "nice": "NICE",
    "happy": "HAPPY",
    "sad": "SAD",
    "angry": "ANGRY",
    "tired": "TIRED",
    "sick": "SICK",
    "big": "BIG",
    "small": "SMALL",
    "little": "SMALL",
    "new": "NEW",
    "old": "OLD",
    "young": "YOUNG",
    "beautiful": "BEAUTIFUL",
    "pretty": "BEAUTIFUL",
    "ugly": "UGLY",
    "easy": "EASY",
    "hard": "HARD",
    "difficult": "HARD",
    "fast": "FAST",
    "quick": "FAST",
    "slow": "SLOW",
    "hot": "HOT",
    "cold": "COLD",
    "hungry": "HUNGRY",
    "thirsty": "THIRSTY",
    "important": "IMPORTANT",
    "right": "RIGHT",
    "wrong": "WRONG",
    "same": "SAME",
    "different": "DIFFERENT",
    "ready": "READY",
    "true": "TRUE",
    "correct": "TRUE",
    "deaf": "DEAF",

    # Adverbs / misc
    "yes": "YES",
    "no": "NO",
    "not": "NOT",
    "never": "NEVER",
    "always": "ALWAYS",
    "sometimes": "SOMETIMES",
    "often": "OFTEN",
    "again": "AGAIN",
    "more": "MORE",
    "also": "ALSO",
    "too": "ALSO",
    "please": "PLEASE",
    "sorry": "SORRY",
    "now": "NOW",
    "later": "LATER",
    "here": "HERE",
    "there": "THERE",
    "maybe": "MAYBE",
    "ok": "OK",
    "okay": "OK",
    "sure": "YES",
    "fine": "FINE",
    "together": "TOGETHER",
    "before": "BEFORE",
    "after": "AFTER",
    "because": "BECAUSE",
    "but": "BUT",
    "and": "AND",
    "or": "OR",
    "if": "IF",
    "then": "THEN",
    "with": "WITH",
    "for": "FOR",
    "from": "FROM",
    "in": "IN",
    "at": "AT",
    "on": "ON",
    "about": "ABOUT",
    "every": "EVERY",
    "all": "ALL",
    "many": "MANY",
    "some": "SOME",
    "enough": "ENOUGH",
    "each": "EACH",
}


def _pos_tag_to_wordnet(tag: str) -> str:
    """Map NLTK POS tag to WordNet POS for the lemmatizer."""
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"  # default noun


def convert_to_sign_gloss(text: str) -> list[str]:
    """
    Convert English text to ASL gloss tokens.

    Pipeline:
      1. Lowercase & strip punctuation
      2. Match multi-word phrases from PHRASE_GLOSSARY
      3. For remaining words: lookup in WORD_GLOSSARY
      4. If not in glossary: lemmatize with NLTK and try again
      5. If still unknown: pass through as uppercase (will be fingerspelled)
      6. Drop filler/grammar words that ASL omits
    """
    # Normalize
    text = text.lower().strip()
    text = re.sub(r"[''']", "'", text)        # normalize apostrophes
    text = re.sub(r"[^\w\s'-]", " ", text)    # strip punctuation except apostrophes/hyphens
    text = re.sub(r"\s+", " ", text).strip()

    # ── Phase 1: Multi-word phrase matching ────────────────────────
    # Replace known phrases with their gloss (longest match first)
    for phrase in sorted(PHRASE_GLOSSARY, key=len, reverse=True):
        if phrase in text:
            replacement = PHRASE_GLOSSARY[phrase]
            text = text.replace(phrase, f" {replacement} ")
            text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    # ── Phase 2: Word-level processing ─────────────────────────────
    # POS-tag for better lemmatization
    tagged = nltk.pos_tag(tokens)

    gloss_tokens: list[str] = []

    for word, tag in tagged:
        # Already converted by phrase matching (uppercase)
        if word.isupper() or "-" in word and word == word.upper():
            # Split multi-token gloss results
            for part in word.split():
                gloss_tokens.append(part)
            continue

        # Skip drop words
        if word in DROP_WORDS:
            continue

        # Skip auxiliaries (ASL mostly drops these)
        if word in AUXILIARIES:
            continue

        # Skip prepositions that ASL drops
        if word in PREPOSITIONS_DROP:
            continue

        # Direct glossary lookup
        if word in WORD_GLOSSARY:
            gloss_tokens.append(WORD_GLOSSARY[word])
            continue

        # Try lemmatization then glossary
        wn_pos = _pos_tag_to_wordnet(tag)
        lemma = _lemmatizer.lemmatize(word, pos=wn_pos)

        if lemma in WORD_GLOSSARY:
            gloss_tokens.append(WORD_GLOSSARY[lemma])
            continue

        # Try verb lemmatization specifically (catches "going" -> "go")
        verb_lemma = _lemmatizer.lemmatize(word, pos="v")
        if verb_lemma in WORD_GLOSSARY:
            gloss_tokens.append(WORD_GLOSSARY[verb_lemma])
            continue

        # Unknown word — pass through uppercase (will be fingerspelled)
        gloss_tokens.append(word.upper())

    return gloss_tokens


# ── Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "Hello, I know you are a good person.",
        "I want to go to the store",
        "She is going to the park later",
        "Can you help me with this?",
        "The weather is nice today",
        "Thank you for your help",
        "I don't understand",
        "What is your name?",
        "How are you?",
        "I'm sorry, I can't come tomorrow",
    ]
    for t in tests:
        result = convert_to_sign_gloss(t)
        print(f"{t:45s} → {' '.join(result)}")
