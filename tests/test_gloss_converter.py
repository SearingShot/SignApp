"""Tests for the hybrid gloss converter."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sign_app.sign_language_text.gloss_converter import convert_to_sign_gloss


def test_basic_greeting():
    result = convert_to_sign_gloss("hello")
    assert result == ["HELLO"], f"Expected ['HELLO'], got {result}"


def test_drops_articles_and_copulas():
    result = convert_to_sign_gloss("I know you are a good person")
    assert "A" not in result, f"Article 'a' should be dropped: {result}"
    # "are" (copula) should be dropped
    for token in result:
        assert token not in ("ARE", "A"), f"Should not contain '{token}': {result}"
    assert "KNOW" in result
    assert "GOOD" in result
    assert "PERSON" in result


def test_verb_lemmatization():
    result = convert_to_sign_gloss("She is going to the park later")
    assert "GO" in result, f"Expected 'GO' from 'going': {result}"
    assert "LATER" in result


def test_contractions():
    result = convert_to_sign_gloss("I don't understand")
    assert "NOT" in result, f"Expected 'NOT' from don't: {result}"
    assert "UNDERSTAND" in result


def test_phrase_matching():
    result = convert_to_sign_gloss("Thank you for your help")
    assert "THANK-YOU" in result, f"Expected 'THANK-YOU': {result}"
    assert "HELP" in result


def test_unknown_word_pasthrough():
    result = convert_to_sign_gloss("I like Rahul")
    assert "RAHUL" in result, f"Unknown word should pass through uppercase: {result}"


def test_question_words():
    result = convert_to_sign_gloss("What is your name?")
    assert "WHAT" in result
    assert "NAME" in result


def test_how_are_you_phrase():
    result = convert_to_sign_gloss("How are you?")
    assert "HOW" in result
    assert "YOU" in result


def test_complex_sentence():
    result = convert_to_sign_gloss("I'm sorry, I can't come tomorrow")
    assert "SORRY" in result
    assert "TOMORROW" in result


def test_empty_string():
    result = convert_to_sign_gloss("")
    assert result == []


def test_all_filler_words():
    result = convert_to_sign_gloss("um uh like well actually")
    assert result == [], f"All filler words should be dropped: {result}"


if __name__ == "__main__":
    tests = [
        test_basic_greeting,
        test_drops_articles_and_copulas,
        test_verb_lemmatization,
        test_contractions,
        test_phrase_matching,
        test_unknown_word_pasthrough,
        test_question_words,
        test_how_are_you_phrase,
        test_complex_sentence,
        test_empty_string,
        test_all_filler_words,
    ]
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")

    print(f"\nRan {len(tests)} tests")
