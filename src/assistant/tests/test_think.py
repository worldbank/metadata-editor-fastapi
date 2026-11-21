from src.assistant.core.think import ThinkSplitter


def run(chunks):
    s = ThinkSplitter()
    out = []
    for c in chunks:
        out += s.feed(c)
    out += s.flush()
    merged = {"reasoning": "", "token": ""}
    for kind, text in out:
        merged[kind] += text
    return merged


def test_plain_text_passes_through():
    assert run(["Hello ", "world"]) == {"reasoning": "", "token": "Hello world"}


def test_tags_split_across_chunks():
    chunks = ["Hel", "lo <th", "ink>hm", "m</t", "hink> wor", "ld"]
    assert run(chunks) == {"reasoning": "hmm", "token": "Hello  world"}


def test_unclosed_think_is_reasoning():
    assert run(["<think>still going"]) == {"reasoning": "still going", "token": ""}


def test_lone_angle_bracket_is_not_swallowed():
    assert run(["a < b", " and c <"]) == {"reasoning": "", "token": "a < b and c <"}


def test_text_is_not_held_longer_than_needed():
    s = ThinkSplitter()
    assert s.feed("abc<t") == [("token", "abc")]
    assert s.feed("able>") == [("token", "<table>")]
