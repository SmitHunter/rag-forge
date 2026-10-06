"""Tests for Gutenberg front-matter cleaning."""

from rag_forge.gutenberg import clean_gutenberg_text, find_body_start

PREFACE_AND_INDEX = """\
*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***

with a Preface by George Saintsbury

Pride and Prejudice is a novel whose opening is often quoted.
The critic discusses pride and the author's first rank.

Chapter I.                                                  1    “He came down”
Heading to Chapter LVI.                                     431
The End                                                     476

[Illustration: ·PRIDE AND PREJUDICE·

Chapter I.]

It is a truth universally acknowledged, that a single man in possession
of a good fortune must be in want of a wife.

However little known the feelings or views of such a man may be on his
first entering a neighbourhood, this truth is so well fixed in the minds
of the surrounding families.

*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***
"""

FRANKENSTEIN_TOC = """\
Frankenstein

CONTENTS
 Letter 1
 Letter 2
 Letter 3
 Letter 4
 Chapter 1
 Chapter 2
 Chapter 3
 Chapter 4

Letter 1

_To Mrs. Saville, England._

St. Petersburgh, Dec. 11th, 17—.

You will rejoice to hear that no disaster has accompanied the
commencement of an enterprise which you have regarded with such evil
forebodings. I arrived here yesterday and my first task is to assure
my dear sister of my welfare and increasing confidence.
"""


def test_strips_wrappers_and_starts_at_chapter_one() -> None:
    cleaned = clean_gutenberg_text(PREFACE_AND_INDEX)

    assert "Saintsbury" not in cleaned
    assert "Heading to Chapter LVI" not in cleaned
    assert "START OF THE PROJECT" not in cleaned
    assert "END OF THE PROJECT" not in cleaned
    assert "It is a truth universally acknowledged" in cleaned
    assert cleaned.lower().index("chapter i") < cleaned.index("It is a truth")
    assert "Illustration" not in cleaned


def test_skips_contents_letter_one() -> None:
    cleaned = clean_gutenberg_text(FRANKENSTEIN_TOC)

    assert cleaned.lstrip().startswith("Letter 1")
    assert "You will rejoice to hear" in cleaned
    assert "CONTENTS" not in cleaned
    assert cleaned.index("St. Petersburgh") < 200


def test_letter_one_crlf_is_not_a_page_number() -> None:
    """A heading of 'Letter 1\\r\\n' must not be mistaken for an index row."""
    text = (
        "Letter 1\r\n\r\nYou will rejoice to hear that no disaster has accompanied "
        "the commencement of an enterprise which you have regarded with such "
        "evil forebodings and my first task is to assure my dear sister of "
        "my welfare.\r\n"
    )
    cleaned = clean_gutenberg_text(text)
    assert cleaned.startswith("Letter 1")
    assert "You will rejoice" in cleaned


def test_find_body_start_returns_zero_without_heading() -> None:
    text = "A short pamphlet with no chapter headings at all. " * 5
    assert find_body_start(text) == 0
    assert "short pamphlet" in clean_gutenberg_text(text)


def test_clean_is_idempotent() -> None:
    once = clean_gutenberg_text(PREFACE_AND_INDEX)
    twice = clean_gutenberg_text(once)
    assert once == twice
