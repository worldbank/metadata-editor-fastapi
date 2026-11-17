"""A document fill sends its batches several at a time to a hosted model, one at a time to a local one."""

import asyncio
from types import SimpleNamespace

from src.assistant.host import docfill

FIELDS = {f"/study_desc/f{i}": {"label": f"Field {i}", "value": "", "found": True, "in_template": True} for i in range(20)}
BUNDLE = {"record_id": "1", "record": {}, "fill": True, "fields": FIELDS}


class SlowGateway:
    def __init__(self, fail_first=0):
        self.running = 0
        self.peak = 0
        self.calls = 0
        self.fail_first = fail_first

    async def complete(self, spec, messages, **_):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("boom")
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.05)
        self.running -= 1
        return "[]"

    async def complete_ex(self, spec, messages, **kw):
        return await self.complete(spec, messages, **kw), "stop", 0

    def mark_failed(self, spec):
        pass


def _run(spec, gateway, **kw):
    return asyncio.run(docfill.run_document_fill(gateway, spec, BUNDLE, "A document.", slot=asyncio.Semaphore(8), **kw))


def _spec(local):
    return SimpleNamespace(alias="m", local=local, max_tokens=4096)


def test_hosted_model_runs_batches_in_parallel_up_to_the_limit(monkeypatch):
    monkeypatch.setattr(docfill, "FILL_PARALLEL", 3)
    gw = SlowGateway()
    out = _run(_spec(False), gw)
    assert gw.calls == 4  # 20 fields in batches of 6
    assert 1 < gw.peak <= docfill.FILL_PARALLEL
    assert out["tool_calls"] == 4
    assert [s["args"]["pointers"][0] for s in out["steps"] if s["stage"] == "call"] == [
        "/study_desc/f0", "/study_desc/f6", "/study_desc/f12", "/study_desc/f18"
    ]  # batch order, not completion order


def test_local_model_runs_one_batch_at_a_time_by_default(monkeypatch):
    monkeypatch.setattr(docfill, "FILL_PARALLEL_LOCAL", 1)
    gw = SlowGateway()
    _run(_spec(True), gw)
    assert gw.peak == 1


def test_local_parallelism_follows_its_own_setting(monkeypatch):
    monkeypatch.setattr(docfill, "FILL_PARALLEL_LOCAL", 2)
    gw = SlowGateway()
    _run(_spec(True), gw)
    assert gw.peak == 2


def test_a_model_that_is_down_stops_the_rest():
    gw = SlowGateway(fail_first=100)
    out = _run(_spec(False), gw)
    assert out["error"]["code"] == "model_error"
    assert gw.calls == 3  # the three in flight failed; the fourth never started
