import pytest

from src.assistant.core.completeness import assess, repair_instruction
from src.assistant.core.tools import ToolArgumentError, calculate, check_args
from src.assistant.host.tools import fit_tool_result

SCHEMA = {
    "type": "object",
    "properties": {"p": {"type": "string", "enum": ["/a", "/b"]}, "n": {"type": "number"}, "f": {"type": "boolean"}},
    "required": ["p"],
}


def test_calculate():
    assert calculate("(2+3)*7") == 35
    assert calculate("-1.5 + .5") == -1
    assert calculate("7/2") == 3.5
    for bad in ["2**3", "import os", "1/0", "(1+2", "1+", "9" * 81]:
        with pytest.raises(ValueError):
            calculate(bad)


def test_check_args():
    assert check_args('{"p": "/a", "n": 2, "f": true}', SCHEMA) == {"p": "/a", "n": 2, "f": True}
    for raw in ['{"n": 1}', '{"p": "/c"}', '{"p": "/a", "x": 1}', '{"p": "/a", "n": "2"}', '{"p": "/a", "f": 1}', "[1]", "not json"]:
        with pytest.raises(ToolArgumentError):
            check_args(raw, SCHEMA)
    with pytest.raises(ToolArgumentError):
        check_args('{"p": "%s"}' % ("x" * 501), {"type": "object", "properties": {"p": {"type": "string"}}})


@pytest.mark.parametrize(
    "text, reason",
    [
        ("", "empty"),
        ("Let me check the abstract field.", "plan-shaped"),
        ("```json\n{", "truncated"),
        ("The title is", None),  # not dangling: "is" is not in the tail list
        ("The fields are title, abstract and", "truncated"),
        ("The title is Household Survey 2020.", None),
        ("Let me know if you need anything else.", None),
        ("Should I check the abstract too?", None),
    ],
)
def test_completeness(text, reason):
    assert assess(text).reason == reason


def test_narrating_detects_echoed_repair():
    instr = repair_instruction(assess("Let me check the universe field."))
    # The quoted part after the first '"' is what a narrating model reads back.
    echoed = f"The user wants me to answer. {instr}"
    assert assess(echoed, instruction=instr).reason == "narrating"


def test_fit_tool_result_trims_variables_then_gives_up():
    page = {"title": "f", "variables": [{"name": f"v{i}", "label": "x" * 100} for i in range(200)]}
    out = fit_tool_result({"view": "data file", "page": page}, 4000)
    assert out["page"]["variablesTruncated"] and 0 < len(out["page"]["variables"]) < 200
    out = fit_tool_result({"view": "p", "record": "r" * 50000}, 4000)
    assert out["truncated"] and len(out["record"]) < 4000
    assert "error" in fit_tool_result({"view": "p", "huge": "z" * 10000}, 4000)
