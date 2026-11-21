import json

from src.assistant.core.toolcalls import parse_calls, strip_calls

S, E = "<|tool_call_start|>", "<|tool_call_end|>"


def args_of(calls):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]


def test_lfm_single_and_multiple_calls():
    text = f"{S}[read_project_field(pointer='/study_desc/study_info/abstract')]{E}"
    assert args_of(parse_calls(text, "lfm")) == [("read_project_field", {"pointer": "/study_desc/study_info/abstract"})]
    text = f"{S}[calculate(expression=\"2*3\"), get_datetime()]{E}"
    assert args_of(parse_calls(text, "lfm")) == [("calculate", {"expression": "2*3"}), ("get_datetime", {})]


def test_lfm_literals_and_escapes():
    text = f"{S}[t(a=1, b=-2.5, c=true, d=None, e='it\\'s', f='\\u00e9')]{E}"
    assert args_of(parse_calls(text, "lfm")) == [("t", {"a": 1, "b": -2.5, "c": True, "d": None, "e": "it's", "f": "é"})]


def test_lfm_is_lenient_about_prose_and_bad_calls():
    text = f"{S}[sure! bad(x=), calculate(expression='1+1') thanks]{E}"
    assert args_of(parse_calls(text, "lfm")) == [("calculate", {"expression": "1+1"})]


def test_truncated_span_is_dropped_not_fatal():
    assert parse_calls(f"{S}[calculate(expression='1+", "lfm") == []
    assert strip_calls(f"Answer. {S}[calculate(expr", "lfm") == "Answer."


def test_hermes_calls():
    text = 'Let me look.<tool_call>{"name": "calculate", "arguments": {"expression": "4/2"}}</tool_call>'
    assert args_of(parse_calls(text, "hermes")) == [("calculate", {"expression": "4/2"})]
    # arguments given as a JSON string, and a malformed block
    text = '<tool_call>{"name": "calculate", "arguments": "{\\"expression\\": \\"1\\"}"}</tool_call><tool_call>{nope</tool_call>'
    assert args_of(parse_calls(text, "hermes")) == [("calculate", {"expression": "1"})]
    assert strip_calls(text, "hermes") == ""


def test_invalid_tool_names_rejected():
    assert parse_calls('<tool_call>{"name": "Bad-Name", "arguments": {}}</tool_call>', "hermes") == []
    assert parse_calls(f"{S}[BadName()]{E}", "lfm") == []
