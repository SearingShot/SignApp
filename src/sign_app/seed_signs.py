"""
Seed MongoDB with ASL sign rules, fingerspelling, handshapes, locations, and movements.

Usage:
    uv run python -m src.sign_app.seed_signs
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

client = MongoClient(MONGODB_URI)
db = client["SignApp"]


# ═══════════════════════════════════════════════════════════
# SIGN RULES — 60+ common ASL signs
# ═══════════════════════════════════════════════════════════

SIGN_RULES = [
    # Greetings & Social
    {"sign": "HELLO",      "handshape": "B",    "location": "forehead",      "movement": "wave"},
    {"sign": "BYE",        "handshape": "OPEN", "location": "neutral_space", "movement": "wave"},
    {"sign": "THANK-YOU",  "handshape": "FLAT", "location": "chin",          "movement": "forward"},
    {"sign": "PLEASE",     "handshape": "FLAT", "location": "chest",         "movement": "circle_clockwise"},
    {"sign": "SORRY",      "handshape": "S",    "location": "chest",         "movement": "circle_clockwise"},
    {"sign": "YES",        "handshape": "S",    "location": "neutral_space", "movement": "nod"},
    {"sign": "NO",         "handshape": "POINT","location": "neutral_space", "movement": "side_to_side"},
    {"sign": "OK",         "handshape": "O",    "location": "neutral_space", "movement": "none"},
    {"sign": "EXCUSE",     "handshape": "FLAT", "location": "chest",         "movement": "forward"},

    # Pronouns
    {"sign": "I",          "handshape": "POINT","location": "chest",         "movement": "tap"},
    {"sign": "YOU",        "handshape": "POINT","location": "neutral_space", "movement": "forward"},
    {"sign": "HE",         "handshape": "POINT","location": "side",          "movement": "forward"},
    {"sign": "SHE",        "handshape": "POINT","location": "side",          "movement": "forward"},
    {"sign": "WE",         "handshape": "POINT","location": "shoulder",      "movement": "circle_clockwise"},
    {"sign": "THEY",       "handshape": "POINT","location": "neutral_space", "movement": "side_to_side"},
    {"sign": "MY",         "handshape": "FLAT", "location": "chest",         "movement": "tap"},
    {"sign": "YOUR",       "handshape": "FLAT", "location": "neutral_space", "movement": "forward"},

    # Common Verbs
    {"sign": "KNOW",       "handshape": "B",    "location": "temple",        "movement": "tap"},
    {"sign": "THINK",      "handshape": "POINT","location": "forehead",      "movement": "tap"},
    {"sign": "WANT",       "handshape": "CLAW", "location": "neutral_space", "movement": "forward"},
    {"sign": "NEED",       "handshape": "POINT","location": "neutral_space", "movement": "nod"},
    {"sign": "LIKE",       "handshape": "OPEN", "location": "chest",         "movement": "forward"},
    {"sign": "LOVE",       "handshape": "FIST", "location": "chest",         "movement": "tap"},
    {"sign": "HELP",       "handshape": "A",    "location": "neutral_space", "movement": "up"},
    {"sign": "SEE",        "handshape": "V",    "location": "nose",          "movement": "forward"},
    {"sign": "LOOK",       "handshape": "V",    "location": "nose",          "movement": "forward"},
    {"sign": "HEAR",       "handshape": "POINT","location": "ear",           "movement": "tap"},
    {"sign": "LISTEN",     "handshape": "C",    "location": "ear",           "movement": "tap"},
    {"sign": "SAY",        "handshape": "POINT","location": "chin",          "movement": "forward"},
    {"sign": "TELL",       "handshape": "POINT","location": "chin",          "movement": "forward"},
    {"sign": "ASK",        "handshape": "POINT","location": "neutral_space", "movement": "forward"},
    {"sign": "GO",         "handshape": "POINT","location": "neutral_space", "movement": "forward"},
    {"sign": "COME",       "handshape": "POINT","location": "neutral_space", "movement": "forward"},
    {"sign": "EAT",        "handshape": "FLAT", "location": "mouth",         "movement": "tap"},
    {"sign": "DRINK",      "handshape": "C",    "location": "mouth",         "movement": "tap"},
    {"sign": "WORK",       "handshape": "S",    "location": "neutral_space", "movement": "tap"},
    {"sign": "LEARN",      "handshape": "FLAT", "location": "forehead",      "movement": "tap"},
    {"sign": "TEACH",      "handshape": "FLAT", "location": "forehead",      "movement": "forward"},
    {"sign": "UNDERSTAND", "handshape": "S",    "location": "temple",        "movement": "tap"},
    {"sign": "FEEL",       "handshape": "OPEN", "location": "chest",         "movement": "up"},
    {"sign": "GIVE",       "handshape": "FLAT", "location": "neutral_space", "movement": "forward"},
    {"sign": "TAKE",       "handshape": "CLAW", "location": "neutral_space", "movement": "forward"},
    {"sign": "MAKE",       "handshape": "S",    "location": "neutral_space", "movement": "twist"},
    {"sign": "GET",        "handshape": "CLAW", "location": "neutral_space", "movement": "forward"},
    {"sign": "HAVE",       "handshape": "B",    "location": "chest",         "movement": "tap"},
    {"sign": "WAIT",       "handshape": "OPEN", "location": "neutral_space", "movement": "none"},
    {"sign": "STOP",       "handshape": "FLAT", "location": "neutral_space", "movement": "down"},
    {"sign": "START",      "handshape": "POINT","location": "neutral_space", "movement": "twist"},
    {"sign": "FINISH",     "handshape": "OPEN", "location": "neutral_space", "movement": "down"},
    {"sign": "TRY",        "handshape": "S",    "location": "neutral_space", "movement": "forward"},
    {"sign": "WALK",       "handshape": "FLAT", "location": "neutral_space", "movement": "forward"},
    {"sign": "SIT",        "handshape": "H",    "location": "neutral_space", "movement": "down"},
    {"sign": "STAND",      "handshape": "V",    "location": "neutral_space", "movement": "none"},
    {"sign": "SIGN",       "handshape": "POINT","location": "neutral_space", "movement": "circle_clockwise"},
    {"sign": "WATCH",      "handshape": "V",    "location": "nose",          "movement": "forward"},
    {"sign": "OPEN",       "handshape": "B",    "location": "neutral_space", "movement": "side_to_side"},
    {"sign": "CLOSE",      "handshape": "B",    "location": "neutral_space", "movement": "forward"},
    {"sign": "BUY",        "handshape": "FLAT", "location": "neutral_space", "movement": "forward"},
    {"sign": "READ",       "handshape": "V",    "location": "neutral_space", "movement": "down"},
    {"sign": "WRITE",      "handshape": "POINT","location": "neutral_space", "movement": "down"},

    # Question Words
    {"sign": "WHAT",       "handshape": "OPEN", "location": "neutral_space", "movement": "side_to_side"},
    {"sign": "WHERE",      "handshape": "POINT","location": "neutral_space", "movement": "side_to_side"},
    {"sign": "WHEN",       "handshape": "POINT","location": "neutral_space", "movement": "circle_clockwise"},
    {"sign": "WHY",        "handshape": "Y",    "location": "forehead",      "movement": "forward"},
    {"sign": "HOW",        "handshape": "FIST", "location": "neutral_space", "movement": "twist"},
    {"sign": "WHO",        "handshape": "L",    "location": "chin",          "movement": "tap"},

    # Nouns
    {"sign": "PERSON",     "handshape": "FLAT", "location": "neutral_space", "movement": "down"},
    {"sign": "PEOPLE",     "handshape": "P",    "location": "neutral_space", "movement": "circle_clockwise"},
    {"sign": "FRIEND",     "handshape": "X",    "location": "neutral_space", "movement": "twist"},
    {"sign": "FAMILY",     "handshape": "F",    "location": "neutral_space", "movement": "circle_clockwise"},
    {"sign": "MOTHER",     "handshape": "OPEN", "location": "chin",          "movement": "tap"},
    {"sign": "FATHER",     "handshape": "OPEN", "location": "forehead",      "movement": "tap"},
    {"sign": "NAME",       "handshape": "H",    "location": "neutral_space", "movement": "tap"},
    {"sign": "HOME",       "handshape": "FLAT", "location": "chin",          "movement": "tap"},
    {"sign": "SCHOOL",     "handshape": "FLAT", "location": "neutral_space", "movement": "double_tap"},
    {"sign": "FOOD",       "handshape": "FLAT", "location": "mouth",         "movement": "tap"},
    {"sign": "WATER",      "handshape": "W",    "location": "chin",          "movement": "tap"},
    {"sign": "MONEY",      "handshape": "FLAT", "location": "neutral_space", "movement": "tap"},
    {"sign": "TIME",       "handshape": "POINT","location": "neutral_space", "movement": "tap"},
    {"sign": "DAY",        "handshape": "D",    "location": "neutral_space", "movement": "down"},
    {"sign": "TODAY",      "handshape": "FLAT", "location": "neutral_space", "movement": "down"},
    {"sign": "TOMORROW",   "handshape": "A",    "location": "chin",          "movement": "forward"},
    {"sign": "YESTERDAY",  "handshape": "A",    "location": "chin",          "movement": "tap"},
    {"sign": "MORNING",    "handshape": "FLAT", "location": "neutral_space", "movement": "up"},
    {"sign": "NIGHT",      "handshape": "FLAT", "location": "neutral_space", "movement": "down"},
    {"sign": "STORE",      "handshape": "FLAT", "location": "neutral_space", "movement": "twist"},

    # Adjectives
    {"sign": "GOOD",       "handshape": "FLAT", "location": "chin",          "movement": "forward"},
    {"sign": "BAD",        "handshape": "FLAT", "location": "chin",          "movement": "down"},
    {"sign": "HAPPY",      "handshape": "FLAT", "location": "chest",         "movement": "circle_clockwise"},
    {"sign": "SAD",        "handshape": "OPEN", "location": "chin",          "movement": "down"},
    {"sign": "BIG",        "handshape": "OPEN", "location": "neutral_space", "movement": "side_to_side"},
    {"sign": "SMALL",      "handshape": "FLAT", "location": "neutral_space", "movement": "tap"},
    {"sign": "BEAUTIFUL",  "handshape": "OPEN", "location": "chin",          "movement": "circle_clockwise"},
    {"sign": "EASY",       "handshape": "FLAT", "location": "neutral_space", "movement": "up"},
    {"sign": "HARD",       "handshape": "V",    "location": "neutral_space", "movement": "tap"},
    {"sign": "HOT",        "handshape": "CLAW", "location": "mouth",         "movement": "forward"},
    {"sign": "COLD",       "handshape": "S",    "location": "neutral_space", "movement": "side_to_side"},
    {"sign": "NEW",        "handshape": "FLAT", "location": "neutral_space", "movement": "forward"},
    {"sign": "OLD",        "handshape": "C",    "location": "chin",          "movement": "down"},
    {"sign": "NICE",       "handshape": "FLAT", "location": "neutral_space", "movement": "forward"},
    {"sign": "FINE",       "handshape": "OPEN", "location": "chest",         "movement": "tap"},
    {"sign": "DIFFERENT",  "handshape": "POINT","location": "neutral_space", "movement": "side_to_side"},
    {"sign": "SAME",       "handshape": "POINT","location": "neutral_space", "movement": "tap"},
    {"sign": "TRUE",       "handshape": "POINT","location": "chin",          "movement": "forward"},
    {"sign": "WRONG",      "handshape": "Y",    "location": "chin",          "movement": "tap"},
    {"sign": "IMPORTANT",  "handshape": "F",    "location": "neutral_space", "movement": "up"},
    {"sign": "READY",      "handshape": "R",    "location": "neutral_space", "movement": "side_to_side"},
    {"sign": "DEAF",       "handshape": "POINT","location": "ear",           "movement": "forward"},
    {"sign": "HUNGRY",     "handshape": "C",    "location": "chest",         "movement": "down"},

    # Adverbs / Misc
    {"sign": "NOT",        "handshape": "A",    "location": "chin",          "movement": "forward"},
    {"sign": "NEVER",      "handshape": "B",    "location": "neutral_space", "movement": "down"},
    {"sign": "ALWAYS",     "handshape": "POINT","location": "neutral_space", "movement": "circle_clockwise"},
    {"sign": "SOMETIMES",  "handshape": "FLAT", "location": "neutral_space", "movement": "tap"},
    {"sign": "AGAIN",      "handshape": "FLAT", "location": "neutral_space", "movement": "tap"},
    {"sign": "MORE",       "handshape": "FLAT", "location": "neutral_space", "movement": "tap"},
    {"sign": "ALSO",       "handshape": "POINT","location": "neutral_space", "movement": "tap"},
    {"sign": "NOW",        "handshape": "FLAT", "location": "neutral_space", "movement": "down"},
    {"sign": "LATER",      "handshape": "L",    "location": "neutral_space", "movement": "forward"},
    {"sign": "HERE",       "handshape": "FLAT", "location": "neutral_space", "movement": "circle_clockwise"},
    {"sign": "THERE",      "handshape": "POINT","location": "neutral_space", "movement": "forward"},
    {"sign": "MAYBE",      "handshape": "FLAT", "location": "neutral_space", "movement": "side_to_side"},
    {"sign": "BECAUSE",    "handshape": "POINT","location": "forehead",      "movement": "forward"},
    {"sign": "BUT",        "handshape": "POINT","location": "neutral_space", "movement": "side_to_side"},
    {"sign": "AND",        "handshape": "OPEN", "location": "neutral_space", "movement": "forward"},
    {"sign": "WITH",       "handshape": "A",    "location": "neutral_space", "movement": "tap"},
    {"sign": "FOR",        "handshape": "POINT","location": "forehead",      "movement": "forward"},
    {"sign": "FROM",       "handshape": "X",    "location": "neutral_space", "movement": "forward"},
    {"sign": "ABOUT",      "handshape": "POINT","location": "neutral_space", "movement": "circle_clockwise"},
    {"sign": "MANY",       "handshape": "S",    "location": "neutral_space", "movement": "forward"},
    {"sign": "ALL",        "handshape": "OPEN", "location": "neutral_space", "movement": "circle_clockwise"},
    {"sign": "EVERY",      "handshape": "A",    "location": "neutral_space", "movement": "down"},
    {"sign": "ENOUGH",     "handshape": "FLAT", "location": "neutral_space", "movement": "forward"},
]


# ═══════════════════════════════════════════════════════════
# FINGERSPELLING
# ═══════════════════════════════════════════════════════════

FINGERSPELLING = [
    {"letter": "A", "handshape": "A", "movement": "none"},
    {"letter": "B", "handshape": "B", "movement": "none"},
    {"letter": "C", "handshape": "C", "movement": "none"},
    {"letter": "D", "handshape": "D", "movement": "none"},
    {"letter": "E", "handshape": "E", "movement": "none"},
    {"letter": "F", "handshape": "F", "movement": "none"},
    {"letter": "G", "handshape": "G", "movement": "none"},
    {"letter": "H", "handshape": "H", "movement": "none"},
    {"letter": "I", "handshape": "I", "movement": "none"},
    {"letter": "J", "handshape": "J", "movement": "circle_clockwise"},
    {"letter": "K", "handshape": "K", "movement": "none"},
    {"letter": "L", "handshape": "L", "movement": "none"},
    {"letter": "M", "handshape": "M", "movement": "none"},
    {"letter": "N", "handshape": "N", "movement": "none"},
    {"letter": "O", "handshape": "O", "movement": "none"},
    {"letter": "P", "handshape": "P", "movement": "none"},
    {"letter": "Q", "handshape": "Q", "movement": "none"},
    {"letter": "R", "handshape": "R", "movement": "none"},
    {"letter": "S", "handshape": "S", "movement": "none"},
    {"letter": "T", "handshape": "T", "movement": "none"},
    {"letter": "U", "handshape": "U", "movement": "none"},
    {"letter": "V", "handshape": "V", "movement": "none"},
    {"letter": "W", "handshape": "W", "movement": "none"},
    {"letter": "X", "handshape": "X", "movement": "none"},
    {"letter": "Y", "handshape": "Y", "movement": "none"},
    {"letter": "Z", "handshape": "Z", "movement": "forward"},
]


# ═══════════════════════════════════════════════════════════
# HANDSHAPES
# ═══════════════════════════════════════════════════════════

HANDSHAPES = [
    {"name": "A"},
    {"name": "B"},
    {"name": "C"},
    {"name": "D"},
    {"name": "E"},
    {"name": "F"},
    {"name": "G"},
    {"name": "H"},
    {"name": "I"},
    {"name": "K"},
    {"name": "L"},
    {"name": "O"},
    {"name": "P"},
    {"name": "R"},
    {"name": "S"},
    {"name": "V"},
    {"name": "W"},
    {"name": "X"},
    {"name": "Y"},
    {"name": "OPEN"},
    {"name": "FIST"},
    {"name": "POINT"},
    {"name": "FLAT"},
    {"name": "CLAW"},
]


# ═══════════════════════════════════════════════════════════
# LOCATIONS
# ═══════════════════════════════════════════════════════════

LOCATIONS = [
    {"name": "neutral_space"},
    {"name": "chest"},
    {"name": "chin"},
    {"name": "mouth"},
    {"name": "nose"},
    {"name": "forehead"},
    {"name": "temple"},
    {"name": "side"},
    {"name": "shoulder"},
    {"name": "ear"},
    {"name": "waist"},
]


# ═══════════════════════════════════════════════════════════
# MOVEMENTS
# ═══════════════════════════════════════════════════════════

MOVEMENTS = [
    {"name": "none"},
    {"name": "tap"},
    {"name": "double_tap"},
    {"name": "circle_clockwise"},
    {"name": "circle_counterclockwise"},
    {"name": "forward"},
    {"name": "down"},
    {"name": "up"},
    {"name": "side_to_side"},
    {"name": "nod"},
    {"name": "twist"},
    {"name": "wave"},
]


def seed():
    """Seed all collections. Uses upsert to avoid duplicates."""
    print("Seeding SignApp database...")

    # Sign rules
    col = db["sign_rules"]
    for rule in SIGN_RULES:
        col.update_one({"sign": rule["sign"]}, {"$set": rule}, upsert=True)
    print(f"  ✓ sign_rules: {len(SIGN_RULES)} signs")

    # Fingerspelling
    col = db["fingerspelling"]
    for fs in FINGERSPELLING:
        col.update_one({"letter": fs["letter"]}, {"$set": fs}, upsert=True)
    print(f"  ✓ fingerspelling: {len(FINGERSPELLING)} letters")

    # Handshapes
    col = db["handshapes"]
    for hs in HANDSHAPES:
        col.update_one({"name": hs["name"]}, {"$set": hs}, upsert=True)
    print(f"  ✓ handshapes: {len(HANDSHAPES)} shapes")

    # Locations
    col = db["locations"]
    for loc in LOCATIONS:
        col.update_one({"name": loc["name"]}, {"$set": loc}, upsert=True)
    print(f"  ✓ locations: {len(LOCATIONS)} locations")

    # Movements
    col = db["movements"]
    for mov in MOVEMENTS:
        col.update_one({"name": mov["name"]}, {"$set": mov}, upsert=True)
    print(f"  ✓ movements: {len(MOVEMENTS)} movements")

    print("\n✅ Database seeded successfully!")


if __name__ == "__main__":
    seed()
