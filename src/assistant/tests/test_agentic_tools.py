"""The editor's agentic tools, called directly (no model): what each returns and what it refuses."""

import asyncio
import json

import pytest

from src.assistant.core.config import parse_registry, ConfigError
from src.assistant.core.tools import ToolArgumentError, ToolContext, check_args
from src.assistant.core.toolcalls import parse_calls
from src.assistant.host.editor import EditorBundle
from src.assistant.host.tools import build_editor_tools

ABSTRACT = "/study_desc/study_info/abstract"
UNIVERSE = "/study_desc/study_info/universe"
KIND = "/study_desc/study_info/data_kind"
KEYWORDS = "/study_desc/study_info/keywords"


def make_bundle(**extra):
    data = {
        "record_id": "6",
        "fill": True,
        "fields": {
            ABSTRACT: {"label": "Abstract", "section": "Study information", "value": "Survey of households.", "required": True},
            UNIVERSE: {"label": "Universe", "section": "Study information", "value": "", "recommended": True, "help": "Who is covered."},
            KIND: {"label": "Kind of data", "section": "Study information", "value": None, "options": ["Sample survey data [ssd]", "Census/enumeration data [cen]"]},
            KEYWORDS: {"label": "Keywords", "section": "Study information", "value": [{"keyword": "income"}], "fillable": False, "type": "array"},
        },
        "datafiles": [{"file_id": "F1", "file_name": "hh.dta", "var_count": 3, "description": ""}],
    }
    data.update(extra)
    return EditorBundle.model_validate(data)


def tools_by_name(bundle, **kw):
    return {t.name: t for t in build_editor_tools(bundle, fill=bundle.fill, **kw)}


def call(tool, args):
    ctx = ToolContext()
    out = asyncio.run(tool.run(check_args(json.dumps(args), tool.parameters), ctx))
    return out, ctx.events


def test_tool_set_follows_the_page():
    names = set(tools_by_name(make_bundle()))
    assert names == {"list_fields", "read_fields", "list_data_files", "propose_field_value"}
    page = {"file_name": "hh.dta", "description": "", "variables": [{"name": "age", "label": "Age"}, {"name": "inc", "label": ""}]}
    names = set(tools_by_name(make_bundle(file_id="F1", datafile=page)))
    assert {"find_variables", "get_current_page", "propose_datafile_description"} <= names
    assert "propose_field_value" not in set(tools_by_name(make_bundle(fill=False)))


def test_list_fields_filters():
    t = tools_by_name(make_bundle())["list_fields"]
    empty, _ = call(t, {"filter": "empty"})
    assert {f["pointer"] for f in empty["fields"]} == {UNIVERSE, KIND}
    req, _ = call(t, {"filter": "required"})
    assert [f["pointer"] for f in req["fields"]] == [ABSTRACT]
    rec, _ = call(t, {"filter": "recommended"})
    assert {f["pointer"] for f in rec["fields"]} == {ABSTRACT, UNIVERSE}
    q, _ = call(t, {"query": "kind data"})
    assert [f["pointer"] for f in q["fields"]] == [KIND]
    allf, _ = call(t, {})
    assert next(f for f in allf["fields"] if f["pointer"] == KEYWORDS)["read_only"] is True


def test_read_fields_batches_and_reports_unknown():
    t = tools_by_name(make_bundle())["read_fields"]
    out, _ = call(t, {"pointers": [ABSTRACT, UNIVERSE, KIND, "/nope"]})
    assert [f["pointer"] for f in out["fields"]] == [ABSTRACT, UNIVERSE, KIND]
    assert out["fields"][1]["help"] == "Who is covered." and out["fields"][1]["empty"] is True
    assert out["fields"][2]["options"][0].startswith("Sample survey")
    assert out["unknown"] == ["/nope"] and "list_fields" in out["hint"]
    many, _ = call(t, {"pointers": [ABSTRACT] + [f"/x{i}" for i in range(20)]})
    assert len(many["not_read_yet"]) == 6 and "again" in many["note"]
    with pytest.raises(ToolArgumentError):
        check_args(json.dumps({"pointers": [str(i) for i in range(61)]}), t.parameters)
    with pytest.raises(ToolArgumentError):
        check_args(json.dumps({"pointers": []}), t.parameters)


def test_proposals_enforce_options_and_fillability():
    t = tools_by_name(make_bundle())["propose_field_value"]
    out, events = call(t, {"pointer": KIND, "value": "sample survey data [SSD]", "reason": "r"})
    assert "proposed" in out and events[0][1]["value"] == "Sample survey data [ssd]"  # the list's own spelling
    out, events = call(t, {"pointer": KIND, "value": "Interviews", "reason": "r"})
    assert "controlled list" in out["error"] and not events
    out, events = call(t, {"pointer": KEYWORDS, "value": "income, poverty", "reason": "r"})
    assert "list or table" in out["error"] and not events
    out, _ = call(t, {"pointer": "/made/up", "value": "x", "reason": "r"})
    assert "list_fields" in out["error"]
    long_value = "x" * 3000  # longer than the default 500-character argument cap
    out, events = call(t, {"pointer": ABSTRACT, "value": long_value, "reason": "r", "status": "improvement"})
    assert events[0][1]["current"] == "Survey of households." and events[0][1]["section"] == "Study information"


def test_proposals_refuse_invented_dates_numbers_and_ids():
    # The reported hallucination: an empty Universe field drafted with a fabricated date.
    t = tools_by_name(make_bundle())["propose_field_value"]
    out, events = call(t, {"pointer": UNIVERSE, "value": "All households surveyed 2025-04-15.", "reason": "r"})
    assert "never states" in out["error"] and "2025-04-15" in out["error"] and not events
    out, events = call(t, {"pointer": UNIVERSE, "value": "Households in HH-2025 and 12000 members.", "reason": "r"})
    assert "HH-2025" in out["error"] and "12000" in out["error"] and not events
    # Words the project does contain are allowed.
    out, events = call(t, {"pointer": UNIVERSE, "value": "All households in the survey.", "reason": "r"})
    assert "proposed" in out and events


def test_proposals_follow_preferred_values():
    bundle = make_bundle(fields={
        UNIVERSE: {"label": "Universe", "value": "", "suggestions": ["% of population", "Percent"]},
    })
    t = tools_by_name(bundle)["propose_field_value"]
    out, events = call(t, {"pointer": UNIVERSE, "value": "Share (per capita)", "reason": "r"})
    assert "% of population" in out["error"] and not events
    out, events = call(t, {"pointer": UNIVERSE, "value": "percent", "reason": "r"})
    assert "proposed" in out and events[0][1]["value"] == "Percent"


def test_proposals_refuse_invented_durations():
    t = tools_by_name(make_bundle())["propose_field_value"]
    out, events = call(t, {"pointer": UNIVERSE, "value": "All households for at least six months.", "reason": "r"})
    assert "six months" in out["error"] and not events
    out, events = call(t, {"pointer": UNIVERSE, "value": "All households in the survey.", "reason": "r"})
    assert "proposed" in out and events


def test_evidence_may_quote_several_fragments():
    t = tools_by_name(make_bundle())["propose_field_value"]
    out, events = call(t, {
        "pointer": UNIVERSE,
        "value": "All households in the survey.",
        "reason": "r",
        "evidence": "Survey of households.; Who is covered.",
    })
    assert "proposed" in out and events


def test_proposals_reject_evidence_not_in_the_project():
    t = tools_by_name(make_bundle())["propose_field_value"]
    out, events = call(t, {"pointer": UNIVERSE, "value": "All households.", "reason": "r", "evidence": "surveyed in April 2025"})
    assert "evidence" in out["error"].lower() and not events


def test_one_proposal_per_field_per_turn():
    t = tools_by_name(make_bundle())["propose_field_value"]
    out, events = call(t, {"pointer": UNIVERSE, "value": "All households.", "reason": "r"})
    assert "proposed" in out and len(events) == 1
    again, events2 = call(t, {"pointer": UNIVERSE, "value": "All households in the survey.", "reason": "r"})
    assert "already proposed" in again["error"] and not events2


def test_datafile_tools():
    page = {"file_name": "hh.dta", "description": "", "variables": [
        {"name": "age", "label": "Age of member"}, {"name": "inc", "label": "", "description": "Monthly income"}]}
    tools = tools_by_name(make_bundle(file_id="F1", datafile=page))
    out, _ = call(tools["find_variables"], {"missing": "label"})
    assert [v["name"] for v in out["variables"]] == ["inc"]
    out, _ = call(tools["find_variables"], {"query": "income"})
    assert out["matches"] == 1
    out, events = call(tools["propose_datafile_description"], {"value": "Household members with age and income.", "reason": "variables"})
    assert events[0][1]["kind"] == "datafile" and events[0][1]["file_id"] == "F1" and events[0][1]["status"] == "empty"


def test_datafile_description_refuses_invented_values():
    page = {"file_name": "hh.dta", "description": "", "variables": [{"name": "age", "label": "Age"}]}
    t = tools_by_name(make_bundle(file_id="F1", datafile=page))["propose_datafile_description"]
    out, events = call(t, {"value": "Households surveyed 2025-04-15.", "reason": "r"})
    assert "2025-04-15" in out["error"] and not events


def test_search_docs_tool_uses_the_host_search():
    async def search(q):
        return ("### Universe — DDI schema guide\nThe population covered.", {"hits": ["Universe — DDI schema guide"]})

    t = tools_by_name(make_bundle(), search_docs=search)["search_docs"]
    out, _ = call(t, {"query": "universe"})
    assert out["found"] and out["sections"] == ["Universe — DDI schema guide"]


def test_lfm_lists_parse():
    calls = parse_calls("<|tool_call_start|>[read_fields(pointers=['/a', \"/b\"])]<|tool_call_end|>", "lfm")
    assert json.loads(calls[0]["function"]["arguments"]) == {"pointers": ["/a", "/b"]}


def test_budget_config_is_validated():
    base = {"alias": "m", "local": True, "litellm_params": {"model": "x"}}
    reg = parse_registry({"models": [base]}, {})
    assert (reg.models["m"].max_tool_calls, reg.models["m"].max_rounds) == (16, 8)
    with pytest.raises(ConfigError):
        parse_registry({"models": [{**base, "max_tool_calls": 100}]}, {})


def test_attached_document_grounds_proposals():
    from src.assistant.host.tools import build_editor_tools

    document = "The survey interviewed 1200 households in 2025."
    bundle = make_bundle()
    tools = {t.name: t for t in build_editor_tools(bundle, fill=True, document=document)}
    out, events = call(tools["propose_field_value"], {"pointer": UNIVERSE, "value": "1200 households in 2025.", "reason": "document"})
    assert "proposed" in out and events, out


def test_list_fields_fillable_filter_matches_what_a_proposal_can_target():
    t = tools_by_name(make_bundle())["list_fields"]
    out, _ = call(t, {"filter": "fillable"})
    ptrs = {f["pointer"] for f in out["fields"]}
    assert KEYWORDS not in ptrs
    assert {ABSTRACT, UNIVERSE, KIND} <= ptrs


def test_list_fields_fillable_filter_excludes_other_templates():
    bundle = make_bundle(fields={
        UNIVERSE: {"label": "Universe", "value": "", "fillable": True},
        "/variable/name": {"label": "Variable name", "value": "", "in_template": False},
    })
    t = tools_by_name(bundle)["list_fields"]
    out, _ = call(t, {"filter": "fillable"})
    assert [f["pointer"] for f in out["fields"]] == [UNIVERSE]
