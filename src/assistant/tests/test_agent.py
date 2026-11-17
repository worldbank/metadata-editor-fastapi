"""Agent loop and editor tools through the route, with a scripted model."""

import json

import pytest

from src.assistant.core import agent as agent_mod
from src.assistant.tests.fakes import ScriptedRouter, native_call_round, text_round
from src.assistant.tests.test_api import events, make_app, post

ABSTRACT = "/study_desc/study_info/abstract"
TITLE = "/study_desc/title_statement/title"
UNIVERSE = "/study_desc/study_info/universe"
# The browser loop's budget, for the tests that exercise the limits themselves.
LEGACY = {"tool_mode": "native", "max_tool_calls": 5, "max_rounds": 5, "max_calls_per_round": 4}


def bundle(**extra):
    b = {
        "record_id": "42",
        "section": "Study description",
        "fields": {
            TITLE: {"label": "Study title", "value": "Household Survey 2020", "in_template": True},
            ABSTRACT: {"label": "Abstract", "value": "A survey of households.", "in_template": True},
            UNIVERSE: {"label": "Universe", "value": None, "found": False, "in_template": True},
            "/series_description/name": {"label": "Indicator name", "value": None, "in_template": False},
        },
        "record": {"study_desc": {"title_statement": {"title": "Household Survey 2020"}}},
    }
    b.update(extra)
    return b


def run(rounds, *, native=True, question="What does the abstract say?", extra=None, **body):
    router = ScriptedRouter(rounds)
    extra = extra or ({"tool_mode": "native"} if native else {"tool_mode": "text", "text_protocol": "lfm"})
    app = make_app(router_obj=router, model_extra=extra)
    body.setdefault("bundle", bundle())
    ev = events(post(app, question=question, **body))
    return ev, router


def of(ev, name):
    return [d for n, d in ev if n == name]


def test_native_tool_round_trip():
    ev, router = run(
        [
            native_call_round(("read_fields", json.dumps({"pointers": [ABSTRACT]}))),
            text_round("The abstract says: ", "A survey of households."),
        ]
    )
    assert of(ev, "meta")[0]["tools"] == ["get_datetime", "calculate", "list_fields", "read_fields", "get_current_page"]
    stages = [(t["stage"], t.get("name")) for t in of(ev, "tool")]
    assert stages == [("call", "read_fields"), ("result", "read_fields")]
    done = of(ev, "done")[0]
    assert done["text"] == "The abstract says: A survey of households."
    assert done["tool_calls"] == 1 and done["rounds"] == 2 and done["call_mode"] == "native"

    first, second = router.calls
    assert first["tool_choice"] == "auto" and first["temperature"] == 0.0
    assert {t["function"]["name"] for t in first["tools"]} >= {"list_fields", "read_fields", "get_current_page"}
    assert "work in steps" in first["messages"][0]["content"]  # agentic instructions
    assistant, tool_msg = second["messages"][-2:]
    assert assistant["tool_calls"][0]["function"]["name"] == "read_fields"
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_0"
    assert json.loads(tool_msg["content"])["fields"][0]["value"] == "A survey of households."


def test_text_protocol_round_trip_hides_markers():
    call = "<|tool_call_start|>[read_fields(pointers=['" + ABSTRACT + "'])]<|tool_call_end|>"
    ev, router = run(
        [text_round("<|tool_call", "_start|>[read_fields(pointers=['", ABSTRACT + "'])]<|tool_call_end|>"), text_round("It describes households.")],
        native=False,
    )
    streamed = "".join(d["text"] for d in of(ev, "token"))
    assert "tool_call" not in streamed and streamed == "It describes households."
    assert of(ev, "done")[0]["call_mode"] == "lfm"
    first, second = router.calls
    assert "tools" not in first
    assert "<|tool_call_start|>" in first["messages"][0]["content"]  # protocol taught in the system prompt
    assert second["messages"][-2] == {"role": "assistant", "content": call}
    last = second["messages"][-1]
    assert last["role"] == "user" and last["content"].startswith("TOOL RESULT read_fields:")


def test_native_server_returning_raw_markers_is_still_parsed():
    router_rounds = [
        text_round("<|tool_call_start|>[calculate(expression='6*7')]<|tool_call_end|>"),
        text_round("42."),
    ]
    # e.g. llama-server started without --jinja: the model's markers come back as plain content.
    ev, router = run(router_rounds, question="what is 6*7", bundle=None, extra={"tool_mode": "native", "text_protocol": "lfm"})
    router_calls = router.calls
    assert of(ev, "done")[0]["text"] == "42."
    assert [t["stage"] for t in of(ev, "tool")] == ["call", "result"]
    assert router_calls[1]["messages"][-1]["content"].startswith("TOOL RESULT calculate:")


def test_tool_budget_and_answer_only_round():
    calls = [native_call_round(("calculate", json.dumps({"expression": f"{i}+1"}))) for i in range(5)]
    ev, router = run([*calls, text_round("Done: the sums were computed.")], question="add things", bundle=None, extra=LEGACY)
    done = of(ev, "done")[0]
    assert done["tool_calls"] == 5
    assert router.calls[-1]["tool_choice"] == "none"  # tools still declared, but not callable
    assert agent_mod.TOOL_BUDGET_SPENT in router.calls[-1]["messages"][-1]["content"]


def test_calls_on_answer_only_round_are_refused_then_repaired():
    calls = [native_call_round(("calculate", json.dumps({"expression": f"{i}*2"}))) for i in range(5)]
    ev, router = run(
        [*calls, native_call_round(("calculate", '{"expression": "9*9"}'), content="Let me check one more."), text_round("Here is the answer.")],
        question="multiply",
        bundle=None,
        extra=LEGACY,
    )
    assert any(t["stage"] == "refused" for t in of(ev, "tool"))
    assert of(ev, "done")[0]["text"] == "Here is the answer."


def test_repeat_call_is_not_rerun():
    args = json.dumps({"pointers": [TITLE]})
    ev, router = run(
        [
            native_call_round(("read_fields", args)),
            native_call_round(("read_fields", args)),
            text_round("The title is Household Survey 2020."),
        ]
    )
    stages = [t["stage"] for t in of(ev, "tool")]
    assert stages == ["call", "result", "repeat"]
    assert of(ev, "done")[0]["tool_calls"] == 1
    assert "already called" in router.calls[2]["messages"][-1]["content"]


def test_bad_arguments_and_unknown_tool_are_fed_back_not_fatal():
    ev, router = run(
        [
            native_call_round(("list_fields", json.dumps({"filter": "bogus"})), ("delete_project", "{}")),
            text_round("I could not read that field."),
        ]
    )
    failed = [t for t in of(ev, "tool") if t["stage"] == "failed"]
    assert [f["name"] for f in failed] == ["list_fields", "delete_project"]
    assert "choose one of" in failed[0]["message"] and "required" in failed[0]["message"]
    assert "no tool named" in failed[1]["message"]
    assert of(ev, "done")[0]["text"] == "I could not read that field."
    tool_msgs = [m for m in router.calls[1]["messages"] if m["role"] == "tool"]
    assert all("error" in json.loads(m["content"]) for m in tool_msgs)


def test_surplus_calls_in_one_round_are_capped():
    many = [("get_datetime", "{}")] + [("calculate", json.dumps({"expression": f"{i}+0"})) for i in range(5)]
    ev, router = run([native_call_round(*many), text_round("Done.")], question="compute", bundle=None, extra=LEGACY)
    assert {"stage": "capped", "dropped": 2} in of(ev, "tool")
    tool_msgs = [m for m in router.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 6  # every call answered, the surplus with an error
    assert "at most 4" in tool_msgs[-1]["content"]


def test_proposals():
    ev, router = run(
        [
            native_call_round(
                ("propose_field_value", json.dumps({"pointer": UNIVERSE, "value": "All households", "reason": "Universe is empty", "status": "empty"})),
                ("propose_field_value", json.dumps({"pointer": "/series_description/name", "value": "x", "reason": "r"})),
                ("propose_field_value", json.dumps({"pointer": TITLE, "value": "  ", "reason": "r"})),
            ),
            text_round("I proposed a universe."),
        ],
        bundle=bundle(fill=True),
    )
    assert "propose_field_value" in of(ev, "meta")[0]["tools"]
    proposals = of(ev, "proposal")
    assert proposals == [
        {"kind": "field", "pointer": UNIVERSE, "label": "Universe", "section": "", "value": "All households",
         "reason": "Universe is empty", "status": "empty", "current": None, "record_id": "42"}
    ]
    tool_msgs = [m for m in router.calls[1]["messages"] if m["role"] == "tool"]
    assert "proposed" in tool_msgs[0]["content"] and "template has no field" in tool_msgs[1]["content"]
    assert "value is empty" in tool_msgs[2]["content"]
    assert "with propose_field_value" in router.calls[0]["messages"][0]["content"]


def test_no_proposal_tool_without_fill():
    ev, _ = run([text_round("Fine.")])
    assert "propose_field_value" not in of(ev, "meta")[0]["tools"]


def test_get_current_page_prefers_datafile():
    page = {"file": "hh.dta", "variables": [{"name": "age", "label": "Age"}]}
    ev, router = run(
        [native_call_round(("get_current_page", "{}")), text_round("The file has an age variable.")],
        bundle=bundle(datafile=page, section="Data file"),
    )
    tool_msg = [m for m in router.calls[1]["messages"] if m["role"] == "tool"][0]
    assert json.loads(tool_msg["content"]) == {"view": "Data file", "page": page}


def test_plan_shaped_reply_is_repaired_once():
    ev, router = run([text_round("Let me check the abstract."), text_round("The abstract describes a household survey.")])
    assert of(ev, "round")[0]["reason"] == "repair"
    done = of(ev, "done")[0]
    assert done["text"] == "The abstract describes a household survey." and done["completeness"]["ok"]
    assert router.calls[1]["messages"][-2] == {"role": "assistant", "content": "Let me check the abstract."}


def test_second_bad_reply_is_reported_incomplete():
    ev, _ = run([text_round("Let me check the abstract."), text_round("")])
    warnings = [w for w in of(ev, "warning") if w["kind"] == "incomplete"]
    assert warnings and warnings[0]["reason"] == "empty"


def test_app_mode_and_greetings():
    ev, _ = run([text_round("Use the Projects page.")], bundle={"section": "Projects"})
    assert of(ev, "meta")[0]["tools"] == ["get_datetime", "calculate"]
    ev, router = run([text_round("Hi! Ask me anything.")], question="hello")
    assert of(ev, "meta")[0]["tools"] == [] and "tools" not in router.calls[0]


def test_slow_tool_times_out(monkeypatch):
    import asyncio

    from src.assistant.core import tools as tools_mod

    async def slow(args, ctx):
        await asyncio.sleep(5)

    monkeypatch.setattr(agent_mod, "TOOL_TIMEOUT_SEC", 0.05)
    original = tools_mod.builtin_tools

    def patched():
        out = original()
        out[0].run = slow
        return out

    monkeypatch.setattr("src.assistant.core.chat.builtin_tools", patched)
    ev, _ = run([native_call_round(("get_datetime", "{}")), text_round("I could not read the time.")], question="time now?", bundle=None)
    failed = [t for t in of(ev, "tool") if t["stage"] == "failed"]
    assert failed and "timed out" in failed[0]["message"]


@pytest.mark.parametrize("native", [True, False])
def test_tool_result_is_capped(native):
    big = {"study_desc": {"abstract": "x" * 60000}}
    call = (
        native_call_round(("get_current_page", "{}"))
        if native
        else text_round("<|tool_call_start|>[get_current_page()]<|tool_call_end|>")
    )
    ev, router = run([call, text_round("It is long.")], native=native, bundle=bundle(record=big))
    msgs = router.calls[1]["messages"]
    payload = msgs[-1]["content"]
    assert len(payload) < 17000


def test_repeated_failing_call_costs_no_second_slot():
    bad = json.dumps({"filter": "bogus"})
    ev, router = run([native_call_round(("list_fields", bad)), native_call_round(("list_fields", bad)), text_round("Could not list.")])
    assert [t["stage"] for t in of(ev, "tool")] == ["failed", "repeat"]
    assert of(ev, "done")[0]["tool_calls"] == 1
