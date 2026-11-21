"""List fields (producers, funders, keywords): row guards and the document fill."""

import asyncio
import json

from src.assistant.core.config import parse_registry
from src.assistant.host.docfill import run_document_fill
from src.assistant.host.listfill import accept_rows, list_targets

DOC = """Study report
Produced by the Statistical Institute of Saint Lucia (SISL) with funding from the World Bank, grant TF-0A1234.
Keywords: household income, poverty, expenditure.
"""
COLUMNS = [{"key": "name", "title": "Name"}, {"key": "abbr", "title": "Abbreviation"}, {"key": "role", "title": "Role"}]


def fields(value=None):
    return {
        "/study_desc/production_statement/producers": {
            "label": "Producers", "section": "Production", "type": "array", "fillable": False, "value": value, "columns": COLUMNS,
        },
        "/study_desc/title_statement/title": {"label": "Title", "type": "string", "fillable": True, "value": ""},
    }


def accept(rows, value=None, evidence="x"):
    f = fields(value)
    ptr = list(f)[0]
    return accept_rows({"field": 1, "rows": rows, "evidence": evidence}, [ptr], f, DOC, DOC, "1")


def test_only_list_fields_with_columns_are_list_targets():
    assert list_targets(fields()) == ["/study_desc/production_statement/producers"]


def test_a_row_is_kept_when_its_main_cell_is_in_the_document():
    card, why, refused = accept([{"name": "Statistical Institute of Saint Lucia", "abbr": "SISL"}])
    assert card["kind"] == "rows" and card["rows"] == [{"name": "Statistical Institute of Saint Lucia", "abbr": "SISL"}]
    assert card["status"] == "empty" and refused == 0


def test_cells_the_document_does_not_carry_are_left_empty_not_invented():
    card, _, _ = accept([{"name": "World Bank", "abbr": "WB", "role": "Funder"}])
    assert card["rows"] == [{"name": "World Bank"}]


def test_a_row_whose_main_cell_is_not_in_the_document_is_refused():
    card, why, refused = accept([{"name": "National Bureau of Invented Facts"}])
    assert card is None and refused == 1


def test_a_row_already_saved_is_not_proposed_again():
    saved = [{"name": "World Bank"}]
    card, why, _ = accept([{"name": "world bank"}, {"name": "Statistical Institute of Saint Lucia"}], saved)
    assert [r["name"] for r in card["rows"]] == ["Statistical Institute of Saint Lucia"]
    assert card["status"] == "improvement" and card["current"] == "1 row saved"


def test_an_invented_identifier_in_a_cell_is_dropped_but_the_row_stays():
    card, _, _ = accept([{"name": "World Bank", "abbr": "TF-9Z9999"}])
    assert card["rows"] == [{"name": "World Bank"}]


def test_columns_the_field_does_not_have_never_reach_the_card():
    card, _, _ = accept([{"name": "World Bank", "colour": "blue"}])
    assert card["rows"] == [{"name": "World Bank"}]


def test_a_generic_key_for_the_main_cell_is_mapped_to_the_main_column():
    f = {"/k": {"label": "Keywords", "type": "array", "value": None, "columns": [{"key": "keyword", "title": "Keyword"}, {"key": "vocab", "title": "Vocabulary"}]}}
    card, _, refused = accept_rows({"field": 1, "rows": [{"name": "poverty"}, {"value": "household income"}]}, ["/k"], f, DOC, DOC, "1")
    assert card["rows"] == [{"keyword": "poverty"}, {"keyword": "household income"}] and refused == 0


def test_duplicate_rows_within_one_answer_collapse():
    card, _, _ = accept([{"name": "World Bank"}, {"name": "World Bank"}])
    assert len(card["rows"]) == 1


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.failed = 0

    async def complete(self, spec, messages, **kw):
        self.prompts.append(messages[0]["content"])
        return self.replies.pop(0)

    async def complete_ex(self, spec, messages, **kw):
        return await self.complete(spec, messages, **kw), "stop", 0

    def mark_failed(self, spec):
        self.failed += 1


def test_document_fill_runs_list_batches_after_the_field_batches():
    reg = parse_registry(
        {"default": "m", "models": [{"alias": "m", "local": True, "litellm_params": {"model": "openai/x", "api_base": "http://x/v1", "api_key": "k"}}]},
        {},
    )
    bundle = {"record_id": "1", "fields": fields()}
    ptr = "/study_desc/production_statement/producers"
    gw = Scripted(
        json.dumps({"proposals": []}),
        json.dumps({"proposals": [{"field": 1, "rows": [{"name": "Statistical Institute of Saint Lucia", "abbr": "SISL"}], "evidence": "Produced by the Statistical Institute of Saint Lucia (SISL)"}]}),
    )
    out = asyncio.run(run_document_fill(gw, reg.resolve("m"), bundle, DOC, slot=asyncio.Semaphore(1)))
    assert "columns:" in gw.prompts[1] and "Tables:" in gw.prompts[1]
    card = out["proposals"][0]
    assert card["kind"] == "rows" and card["pointer"] == ptr
    assert "**1** new table row" in out["text"]
    assert [s["name"] for s in out["steps"] if s["stage"] == "call"] == ["fill_from_document", "fill_lists_from_document"]
