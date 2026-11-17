"""A fill batch with no usable reply is retried and, if it stays unusable, reported as not checked."""

import asyncio
import json
from types import SimpleNamespace

from src.assistant.host import docfill
from src.assistant.host.docfill import parse_proposals_strict

GOOD = json.dumps({"proposals": [{"field": 1, "value": "Household Income Survey 2025", "evidence": "Household Income Survey 2025", "reason": "stated"}]})


def test_strict_parser_tells_nothing_to_propose_from_no_answer():
    assert parse_proposals_strict("") is None
    assert parse_proposals_strict("   ") is None
    assert parse_proposals_strict('{"proposals": []}') == []
    assert parse_proposals_strict("[]") == []
    assert parse_proposals_strict("I could not find anything.") is None
    assert len(parse_proposals_strict("Here you go:\n```json\n" + GOOD + "\n```")) == 1


def test_a_reply_cut_off_part_way_keeps_the_complete_proposals():
    cut = '{"proposals": [{"field": 1, "value": "A", "evidence": "a"}, {"field": 2, "value": "B", "evidence": "b"}, {"field": 3, "value": "C", "evid'
    kept = parse_proposals_strict(cut)
    assert [p["field"] for p in kept] == [1, 2]
    assert parse_proposals_strict('{"proposals": [{"field": 1, "value": "A", "evid') is None


class ScriptedGateway:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.failed = 0

    async def complete_ex(self, spec, messages, **kw):
        self.calls.append((messages[0]["content"], kw["max_tokens"]))
        return (self.replies.pop(0) if self.replies else ""), "length", 9000

    def mark_failed(self, spec):
        self.failed += 1


FIELDS = {"/study_desc/title": {"label": "Title", "value": "", "found": True, "in_template": True}}
BUNDLE = {"record_id": "1", "record": {}, "fill": True, "fields": FIELDS}
SPEC = SimpleNamespace(alias="m", local=True, max_tokens=6144)


def run(gateway):
    return asyncio.run(docfill.run_document_fill(gateway, SPEC, BUNDLE, "Household Income Survey 2025.", slot=asyncio.Semaphore(4)))


def test_a_hosted_model_is_asked_plainly_first_and_a_local_one_is_told_to_be_brief_from_the_start():
    hosted = ScriptedGateway(['{"proposals": []}'])
    asyncio.run(docfill.run_document_fill(hosted, SimpleNamespace(alias="h", local=False, max_tokens=6144), BUNDLE, "Doc.", slot=asyncio.Semaphore(4)))
    assert docfill.BREVITY_NUDGE not in hosted.calls[0][0]
    local = ScriptedGateway(['{"proposals": []}'])
    run(local)
    assert docfill.BREVITY_NUDGE in local.calls[0][0]


def test_an_empty_first_reply_is_retried_with_a_nudge_and_the_full_budget():
    gw = ScriptedGateway(["", GOOD])
    out = run(gw)
    assert [p["pointer"] for p in out["proposals"]] == ["/study_desc/title"]
    assert len(gw.calls) == 2
    assert docfill.BREVITY_NUDGE in gw.calls[1][0]
    assert gw.calls[1][1] == SPEC.max_tokens
    assert gw.failed == 0


def test_when_no_batch_gets_a_usable_reply_the_fill_says_the_model_did_not_answer():
    gw = ScriptedGateway(["", ""])
    out = run(gw)
    assert out["error"]["code"] == "model_error"
    assert gw.failed == 0  # an unusable reply is not an outage: the model is not marked down


def test_one_unusable_batch_is_reported_as_not_checked_and_the_others_still_count():
    many = {f"/study_desc/f{i}": {"label": f"Field {i}", "value": "", "found": True, "in_template": True} for i in range(8)}
    gw = ScriptedGateway(["", "", GOOD])  # batch 1: empty twice; batch 2 (two fields): fine
    out = asyncio.run(docfill.run_document_fill(gw, SPEC, {"record_id": "1", "record": {}, "fill": True, "fields": many}, "Household Income Survey 2025.", slot=asyncio.Semaphore(4)))
    assert len(out["proposals"]) == 1
    assert "1 batch of fields could not be checked" in out["text"]
    assert gw.failed == 0


def test_a_valid_empty_answer_is_nothing_to_propose_and_is_not_retried():
    gw = ScriptedGateway(['{"proposals": []}'])
    out = run(gw)
    assert len(gw.calls) == 1
    assert "could not be checked" not in out["text"]
