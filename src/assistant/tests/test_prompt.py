from src.assistant.core.config import ModelSpec
from src.assistant.core.prompt import (
    CONTEXT_SEPARATOR,
    build_messages,
    byte_cap,
    compose_context,
    routes_direct,
    system_prompt,
)

SPEC = ModelSpec(alias="m", label="M", local=True, litellm_params={"model": "x"}, context_window=1024, max_tokens=200)


def chars(spec, messages):
    # 1 token per 4 characters: deterministic stand-in for a tokenizer.
    return sum(len(m["content"]) for m in messages) // 4


def test_routes_direct():
    assert routes_direct("hello")
    assert routes_direct("What can you do?")
    assert not routes_direct("What is the abstract of this study?")
    assert not routes_direct("hello " * 40)


def test_host_context_is_appended_never_replacing_the_default():
    text = compose_context(SPEC, "## About this application\nhelp")
    assert text.startswith("About this assistant")
    assert CONTEXT_SEPARATOR in text and text.endswith("help")
    assert compose_context(SPEC, "  ") == compose_context(SPEC, "")


def test_default_context_says_where_the_model_runs():
    hosted = ModelSpec(alias="h", label="H", local=False, litellm_params={"model": "x"})
    assert "external model provider" in compose_context(hosted, "")
    assert "external model provider" not in compose_context(SPEC, "")


def test_system_prompt_modes_differ():
    assert "saved metadata" in system_prompt("Editor", "record")
    assert "help text" in system_prompt("Editor", "app")


def test_messages_shape_and_history_limit():
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    a = build_messages(SPEC, system="S", context="C", history=hist, question="Q", max_history=6, max_tokens=200, count=chars)
    assert a.messages[0]["role"] == "system" and "C" in a.messages[0]["content"]
    assert [m["content"] for m in a.messages[1:]] == ["m4", "m5", "m6", "m7", "m8", "m9", "Q"]
    assert a.dropped_messages == 0


def test_history_never_starts_with_an_answer():
    hist = [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}, {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    a = build_messages(SPEC, system="S", context="C", history=hist, question="Q", max_history=3, max_tokens=200, count=chars)
    assert a.messages[1]["role"] == "user"


def test_window_fit_drops_oldest_exchanges_then_cuts_context():
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 800} for i in range(6)]
    a = build_messages(SPEC, system="S", context="C" * 400, history=hist, question="Q", max_history=6, max_tokens=200, count=chars)
    assert a.prompt_tokens <= SPEC.context_window - 200
    assert a.dropped_messages >= 2 and a.dropped_messages % 2 == 0
    assert "dropped to fit" in a.messages[0]["content"]

    big = build_messages(SPEC, system="S", context="C" * 10000, history=[], question="Q", max_history=6, max_tokens=200, count=chars)
    assert big.context_truncated and big.prompt_tokens <= SPEC.context_window - 200


def test_byte_cap_keeps_utf8_valid():
    text, cut = byte_cap("é" * 10, 5)
    assert cut and text.startswith("éé") and "�" not in text
    assert byte_cap("short", 100) == ("short", False)
