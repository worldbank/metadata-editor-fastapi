"""Reviews and fills share a few job slots, so chat keeps model slots free."""

import asyncio
import json

from fastapi.testclient import TestClient

from src.assistant.tests.fakes import ScriptedRouter, native_call_round, text_round
from src.assistant.tests.test_api import make_app, wait_for_job
from src.routers.assistant import _take_job_slot

PROPOSE = ("propose_field_value", json.dumps({"pointer": "/study_desc/title", "value": "Household survey", "reason": "record says so"}))
BUNDLE = {
    "record_id": "12",
    "record": {"study_desc": {"title": ""}},
    "fill": True,
    "fields": {"/study_desc/title": {"label": "Title", "value": "", "found": True, "in_template": True}},
}


class _App:
    pass


def test_a_job_waits_for_a_job_slot_and_gives_it_back():
    async def scenario():
        app = _App()
        app.assistant_job_sem = asyncio.Semaphore(1)
        first = await _take_job_slot(app, lambda: False)
        assert first is app.assistant_job_sem
        second = asyncio.create_task(_take_job_slot(app, lambda: False))
        await asyncio.sleep(0.05)
        assert not second.done()  # the only job slot is taken
        first.release()
        got = await asyncio.wait_for(second, timeout=3)
        assert got is app.assistant_job_sem
        got.release()

    asyncio.run(scenario())


def test_a_job_cancelled_while_it_waits_gives_up():
    async def scenario():
        app = _App()
        app.assistant_job_sem = asyncio.Semaphore(1)
        held = await _take_job_slot(app, lambda: False)
        result = await _take_job_slot(app, lambda: True)
        assert result is False
        held.release()

    asyncio.run(scenario())


def test_without_a_job_semaphore_nothing_is_limited():
    assert asyncio.run(_take_job_slot(_App(), lambda: False)) is None


def test_two_reviews_with_one_job_slot_both_finish(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "jobs").mkdir()
    rounds = [native_call_round(PROPOSE), text_round("First done."), native_call_round(PROPOSE), text_round("Second done.")]
    app = make_app(router_obj=ScriptedRouter(rounds), model_extra={"tool_mode": "native"})
    app.jobs = {}
    app.assistant_job_sem = asyncio.Semaphore(1)
    with TestClient(app) as client:
        ids = [client.post("/assistant/review/jobs", json={"bundle": BUNDLE}).json()["job_id"] for _ in range(2)]
        for jobid in ids:
            wait_for_job(app, jobid)
    assert [app.jobs[j]["status"] for j in ids] == ["done", "done"]
    # Back to a free slot afterwards: nothing leaked.
    assert app.assistant_job_sem._value == 1
