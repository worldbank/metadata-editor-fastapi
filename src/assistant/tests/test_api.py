"""Route-level tests: real LiteLLM Router with mock_response, no network, no model server."""

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.assistant.core.chat import TurnLimits
from src.assistant.core.config import parse_registry
from src.assistant.core.llm import Gateway
from src.assistant.host.editor import EditorHost
from src.routers.assistant import router


def make_app(mock_response="Hello <think>checking</think>world", *, router_obj=None, help_path=None, gateway=True, model_extra=None):
    reg = parse_registry(
        {
            "default": "local",
            "models": [
                {
                    "alias": "local",
                    "label": "Local test",
                    "local": True,
                    "litellm_params": {
                        "model": "openai/test",
                        "api_base": "http://127.0.0.1:9/v1",
                        "api_key": "none",
                        "mock_response": mock_response,
                    },
                    "context_window": 4096,
                    "max_tokens": 256,
                    **(model_extra or {}),
                },
                {"alias": "hosted", "local": False, "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "${OPENAI_API_KEY}"}},
            ],
        },
        {},
    )
    app = FastAPI()
    app.include_router(router)
    app.assistant_gateway = Gateway(reg, router=router_obj) if gateway else None
    app.assistant_config_error = None if gateway else "model registry not found: assistant.models.yaml"
    app.assistant_sem = asyncio.Semaphore(2)
    app.assistant_max_inflight = 4
    app.assistant_inflight = 0
    app.assistant_limits = TurnLimits()
    app.assistant_host = EditorHost(help_path)
    return app


def events(response):
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def post(app, **body):
    body.setdefault("question", "What does the abstract field hold?")
    with TestClient(app) as client:
        return client.post("/assistant/chat/stream", json=body)


def wait_for_job(app, jobid, timeout=5.0):
    """Jobs run as tasks on the TestClient's loop (another thread); wait while the client is open."""
    import time

    deadline = time.monotonic() + timeout
    while app.jobs[jobid]["status"] in ("queued", "processing") and time.monotonic() < deadline:
        time.sleep(0.02)


def test_stream_event_sequence_and_reasoning_split(tmp_path):
    help_file = tmp_path / "help.md"
    help_file.write_text("The Projects page lists every project.", encoding="utf-8")
    r = post(make_app(help_path=help_file), bundle={"record_id": "12", "section": "Study description"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = events(r)
    names = [n for n, _ in ev]
    assert names[0] == "meta" and names[-1] == "done"
    assert "error" not in names

    meta = ev[0][1]
    assert meta["model"] == "local" and meta["local"] is True and meta["mode"] == "record"
    assert meta["sources"] == ["help", "view"]

    tokens = "".join(d["text"] for n, d in ev if n == "token")
    reasoning = "".join(d["text"] for n, d in ev if n == "reasoning")
    assert tokens.replace(" ", "") == "Helloworld" and reasoning == "checking"

    done = ev[-1][1]
    assert done["text"] == tokens.strip()
    assert done["finish_reason"] == "stop"
    assert "The Projects page lists every project." in done["sent"]["context"]
    assert done["sent"]["context"].startswith("About this assistant")
    assert "saved metadata" in done["sent"]["system"]


def test_attached_document_reaches_the_turn_context():
    app = make_app()
    r = post(app, bundle={"record_id": "12"}, document="The survey covers 1,200 households in 2025.")
    done = [d for n, d in events(r) if n == "done"][0]
    assert "Attached document" in done["sent"]["context"]
    assert "1,200 households" in done["sent"]["context"]


def test_app_mode_without_record():
    ev = events(post(make_app()))
    assert ev[0][1]["mode"] == "app"


def test_greeting_routes_direct():
    ev = events(post(make_app(), question="hello"))
    assert ev[0][1]["direct"] is True
    assert "greeted you" in ev[-1][1]["sent"]["system"]


def test_unknown_or_unconfigured_model_is_rejected_before_streaming():
    app = make_app()
    assert post(app, model="nope").status_code == 400
    assert post(app, model="hosted").status_code == 400  # OPENAI_API_KEY unset


def test_wrong_protocol_is_rejected():
    assert post(make_app(), protocol="pslm-server/0").status_code == 400


def test_unconfigured_assistant_returns_503_and_status_explains():
    app = make_app(gateway=False)
    assert post(app).status_code == 503
    with TestClient(app) as client:
        status = client.get("/assistant/status").json()
    assert status["enabled"] is False and "not found" in status["error"]


def test_inflight_limit():
    app = make_app()
    app.assistant_inflight = app.assistant_max_inflight
    assert post(app).status_code == 429


class FailingRouter:
    async def acompletion(self, **kwargs):
        raise RuntimeError("connection refused\nsecret details on the next line")


def test_model_failure_ends_with_one_error_event():
    app = make_app(router_obj=FailingRouter())
    app.assistant_gateway._probe_cache["local"] = (10**12, True)
    ev = events(post(app))
    assert [n for n, _ in ev] == ["meta", "error"]
    assert ev[-1][1]["code"] == "model_error"
    assert "secret details" not in ev[-1][1]["message"]
    with TestClient(app) as client:
        assert client.get("/assistant/models").json()["models"][0]["available"] is False


class Chunk:
    def __init__(self, content=None, finish=None, model="m"):
        delta = type("D", (), {"content": content, "reasoning_content": None})()
        choice = type("C", (), {"delta": delta, "finish_reason": finish})()
        self.choices = [choice]
        self.model = model
        self.usage = None


class LengthRouter:
    async def acompletion(self, **kwargs):
        async def gen():
            yield Chunk("partial answer")
            yield Chunk(None, finish="length")

        return gen()


def test_token_cap_warning():
    ev = events(post(make_app(router_obj=LengthRouter())))
    kinds = [d["kind"] for n, d in ev if n == "warning"]
    assert kinds == ["token_cap"]
    assert ev[-1][1]["text"] == "partial answer"


class EmptyRouter:
    async def acompletion(self, **kwargs):
        async def gen():
            yield Chunk("<think>only thinking</think>")
            yield Chunk(None, finish="stop")

        return gen()


def test_reasoning_only_reply_is_flagged_incomplete():
    ev = events(post(make_app(router_obj=EmptyRouter())))
    assert [d["kind"] for n, d in ev if n == "warning"] == ["incomplete"]
    assert ev[-1][1]["text"] == ""


class CapturingRouter:
    def __init__(self):
        self.kwargs = None

    async def acompletion(self, **kwargs):
        self.kwargs = kwargs

        async def gen():
            yield Chunk("ok", finish="stop")

        return gen()


def test_request_max_tokens_cannot_exceed_model_limit():
    cap = CapturingRouter()
    post(make_app(router_obj=cap), max_tokens=10000)
    assert cap.kwargs["max_tokens"] == 256
    assert cap.kwargs["model"] == "local" and cap.kwargs["stream"] is True


def test_models_endpoint_hides_endpoints_and_keys():
    with TestClient(make_app()) as client:
        body = client.get("/assistant/models").json()
    assert [m["alias"] for m in body["models"]] == ["local"]
    assert body["models"][0]["default"] is True
    assert body["models"][0]["available"] is False  # nothing listens on 127.0.0.1:9
    assert "127.0.0.1" not in json.dumps(body) and "openai/test" not in json.dumps(body)


@pytest.mark.parametrize("bad", [{"record_id": ["not", "a", "string"]}, {"record_id": "x" * 500}])
def test_malformed_bundle_is_ignored_not_fatal(bad):
    r = post(make_app(), bundle=bad)
    assert r.status_code == 200 and events(r)[0][1]["mode"] == "app"


def test_suggest_datafile_description_returns_parsed_suggestion():
    app = make_app(mock_response='{"suggestion": "A file of households.", "reason": "Variables names include HHID."}')
    with TestClient(app) as client:
        r = client.post(
            "/assistant/suggest",
            json={"task": "datafile-description", "bundle": {"record_id": "12", "datafile": {"file_name": "h.dta", "variables": [{"name": "HHID"}]}}, "protocol": "pslm-server/1"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "datafile-description"
    assert body["suggestion"] == "A file of households."
    assert body["reason"] == "Variables names include HHID."


def test_suggest_resource_field_uses_the_field_pointer():
    app = make_app(mock_response='Here is a better universe: households with children.')
    with TestClient(app) as client:
        r = client.post(
            "/assistant/suggest",
            json={
                "task": "resource-field",
                "field": "/study_desc/study_info/universe",
                "bundle": {"record_id": "12", "record": {}, "fields": {"/study_desc/study_info/universe": {"label": "Universe", "value": ""}}},
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "resource-field"
    # No JSON in the reply: the raw text becomes the suggestion rather than an empty one.
    assert body["suggestion"].startswith("Here is a better universe")


def test_documents_extract_reads_text_and_rejects_unknown_types(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_PATH", "")
    import src.utils.path_security as ps

    monkeypatch.setattr(ps, "PATH_VALIDATION_ENABLED", False)
    monkeypatch.setattr(ps, "STORAGE_PATH", None)
    doc = tmp_path / "notes.md"
    doc.write_text("# Report\n\nHousehold survey.", encoding="utf-8")
    binary = tmp_path / "image.exe"
    binary.write_bytes(b"\x00\x01")
    app = make_app()
    with TestClient(app) as client:
        ok = client.post("/assistant/documents/extract", json={"file_path": str(doc)})
        bad = client.post("/assistant/documents/extract", json={"file_path": str(binary)})
    assert ok.status_code == 200 and "Household survey" in ok.json()["text"]
    assert bad.status_code == 400 and "Cannot read" in bad.json()["detail"]


def test_review_job_runs_the_turn_and_stores_proposals(tmp_path, monkeypatch):
    from src.assistant.tests.fakes import ScriptedRouter, native_call_round, text_round

    monkeypatch.chdir(tmp_path)
    (tmp_path / "jobs").mkdir()
    rounds = [
        native_call_round(
            ("propose_field_value", json.dumps({"pointer": "/study_desc/title", "value": "Household survey", "reason": "record says so"}))
        ),
        text_round("Reviewed: one field proposed."),
    ]
    app = make_app(router_obj=ScriptedRouter(rounds), model_extra={"tool_mode": "native"})
    app.jobs = {}
    bundle = {
        "record_id": "12",
        "record": {"study_desc": {"title": ""}},
        "fill": True,
        "fields": {"/study_desc/title": {"label": "Title", "value": "", "found": True, "in_template": True}},
    }
    with TestClient(app) as client:
        r = client.post("/assistant/review/jobs", json={"bundle": bundle})
        assert r.status_code == 200 and r.json()["status"] == "queued"
        jobid = r.json()["job_id"]
        assert app.jobs[jobid]["jobtype"] == "assistant-review"
        wait_for_job(app, jobid)
    assert app.jobs[jobid]["status"] == "done"
    stored = json.loads((tmp_path / "jobs" / f"{jobid}.json").read_text(encoding="utf-8"))
    assert stored["text"] == "Reviewed: one field proposed."
    assert stored["proposals"][0]["pointer"] == "/study_desc/title"
    assert stored["proposals"][0]["value"] == "Household survey"


def test_suggest_resource_field_includes_the_attached_document():
    app = make_app(mock_response='{"suggestion": "Households in the survey year.", "reason": "Title says so."}')
    with TestClient(app) as client:
        r = client.post(
            "/assistant/suggest",
            json={
                "task": "resource-field",
                "field": "/study_desc/study_info/universe",
                "document": "All households interviewed in 2025.",
                "bundle": {"record_id": "12", "record": {}, "fields": {"/study_desc/study_info/universe": {"label": "Universe", "value": ""}}},
            },
        )
    assert r.status_code == 200 and r.json()["suggestion"].startswith("Households")


def test_resource_field_prompt_carries_the_document_and_the_saved_value():
    from src.assistant.host.tasks import resource_field_messages

    messages = resource_field_messages(
        {
            "record_id": "12",
            "record": {},
            "fields": {"/u": {"label": "Universe", "value": "All persons", "options": ["All persons", "Children"]}},
        },
        "/u",
        "Households interviewed in 2025.",
    )
    prompt = messages[0]["content"]
    assert "Households interviewed in 2025." in prompt
    assert "All persons" in prompt and "Children" in prompt
    assert "Field label: Universe" in prompt


def test_inflight_count_returns_to_zero_after_a_turn_and_after_a_failure():
    app = make_app()
    post(app)
    assert app.assistant_inflight == 0

    failing = make_app(router_obj=FailingRouter())
    failing.assistant_gateway._probe_cache["local"] = (10**12, True)
    post(failing)
    assert failing.assistant_inflight == 0


def test_cancellation_stops_the_stream_between_deltas():
    import asyncio

    from src.assistant.core.llm import Gateway
    from src.assistant.tests.fakes import ScriptedRouter, text_round

    reg = parse_registry(
        {
            "default": "local",
            "models": [
                {
                    "alias": "local",
                    "local": True,
                    "litellm_params": {"model": "openai/test", "api_base": "http://127.0.0.1:9/v1", "api_key": "none"},
                    "context_window": 4096,
                    "max_tokens": 256,
                }
            ],
        },
        {},
    )
    gateway = Gateway(reg, router=ScriptedRouter([text_round("one ", "two ", "three ")]))
    calls = {"n": 0}

    def cancelled():
        calls["n"] += 1
        return calls["n"] > 1

    async def run():
        out = []
        async for d in gateway.stream(reg.resolve("local"), [{"role": "user", "content": "hi"}], max_tokens=64, temperature=0, is_cancelled=cancelled):
            out.append(d)
        return out

    deltas = asyncio.run(run())
    finish = [d for d in deltas if d.kind == "finish"][0]
    assert finish.finish_reason == "cancelled"
    assert finish.content == "one"  # the chunk already read, nothing after the cancel check


def test_suggest_resource_field_rejects_unknown_pointer():
    app = make_app(mock_response='{"suggestion": "x", "reason": "y"}')
    with TestClient(app) as client:
        r = client.post(
            "/assistant/suggest",
            json={
                "task": "resource-field",
                "field": "/made/up",
                "bundle": {"record_id": "12", "fields": {"/study_desc/study_info/universe": {"label": "Universe", "value": ""}}},
            },
        )
    assert r.status_code == 400
    assert "not a field of this project" in r.json()["detail"]


def test_suggest_resource_field_rejects_a_list_field():
    app = make_app(mock_response='{"suggestion": "x", "reason": "y"}')
    with TestClient(app) as client:
        r = client.post(
            "/assistant/suggest",
            json={
                "task": "resource-field",
                "field": "/study_desc/study_info/keywords",
                "bundle": {"record_id": "12", "fields": {"/study_desc/study_info/keywords": {"label": "Keywords", "value": [], "fillable": False}}},
            },
        )
    assert r.status_code == 400
    assert "list or table" in r.json()["detail"]


def test_suggest_normalises_and_enforces_preferred_values():
    app = make_app(mock_response='{"suggestion": "percent", "reason": "the series name says % of population"}')
    body = {
        "task": "resource-field",
        "field": "/series_description/measurement_unit",
        "bundle": {"record_id": "12", "fields": {
            "/series_description/measurement_unit": {"label": "Measurement unit", "value": "", "suggestions": ["% of population", "Percent"]}
        }},
    }
    with TestClient(app) as client:
        ok = client.post("/assistant/suggest", json=body)
    assert ok.status_code == 200 and ok.json()["suggestion"] == "Percent"

    app = make_app(mock_response='{"suggestion": "Share (per capita)", "reason": "guess"}')
    with TestClient(app) as client:
        bad = client.post("/assistant/suggest", json=body)
    assert bad.status_code == 200
    assert bad.json()["suggestion"] == ""
    assert "% of population" in bad.json()["message"]


# ---------------------------------------------------------------- attached documents and document fill

LONG_DOC = "\n\n".join(
    ["# Household Income Survey 2025\n\nNational Statistics Office."]
    + [f"Chapter {n}. Field operations notes, interviewer training and logistics, part {n}." * 3 for n in range(40)]
    + ["Sampling procedure: a two-stage stratified cluster sample of 1,200 households was drawn."]
)


def test_long_document_excerpt_keeps_the_opening_and_the_matching_passage():
    from src.assistant.core.documents import AttachedDocument

    doc = AttachedDocument(LONG_DOC)
    text, cut = doc.excerpt("What was the sampling procedure?", 2000)
    assert cut and len(text.encode("utf-8")) <= 2000
    assert text.startswith("Household Income Survey 2025\nNational Statistics Office.")
    assert "two-stage stratified cluster sample" in text  # the last passage, far past a head cut
    short, cut = AttachedDocument("One line.").excerpt("anything", 2000)
    assert short == "One line." and not cut


def test_search_document_tool_is_offered_with_an_attachment():
    from src.assistant.core.tools import ToolContext
    from src.assistant.host.editor import parse_bundle
    from src.assistant.host.tools import build_editor_tools

    bundle = parse_bundle({"record_id": "12", "fields": {"/t": {"label": "Title", "value": ""}}})
    assert "search_document" not in [t.name for t in build_editor_tools(bundle, fill=True)]
    tools = {t.name: t for t in build_editor_tools(bundle, fill=True, document=LONG_DOC)}
    result = asyncio.run(tools["search_document"].run({"query": "sampling procedure"}, ToolContext()))
    assert result["found"] and "1,200 households" in result["passages"][0]["text"]


FILL_FIELDS = {
    "/study_desc/title": {"label": "Title", "value": "", "required": True, "fillable": True},
    "/study_desc/sampling": {"label": "Sampling procedure", "value": "Random sample.", "fillable": True},
    "/study_desc/kind": {"label": "Kind of data", "value": "", "options": ["Sample survey data", "Census data"]},
    "/study_desc/series": {"label": "Series", "value": "", "fillable": False},
}


def test_fill_targets_put_empty_required_fields_first_and_skip_unfillable_ones():
    from src.assistant.host.docfill import BATCH, fill_targets

    fields = dict(FILL_FIELDS)
    fields.update({f"/x/{n}": {"label": f"Extra {n}", "value": "set"} for n in range(100)})
    targets = fill_targets(fields)
    assert "/study_desc/series" not in targets
    assert {"/study_desc/title", "/study_desc/kind", "/study_desc/sampling"} <= set(targets[:BATCH])


def test_document_fill_proposes_grounded_values_and_drops_the_rest():
    from src.assistant.host.docfill import accept, parse_proposals

    pointers = list(FILL_FIELDS)
    doc = "Household Income Survey 2025. A two-stage stratified cluster sample. Sample survey data."
    reply = """```json
    {"proposals": [
      {"field": 1, "value": "Household Income Survey 2025", "evidence": "Household Income Survey 2025", "reason": "Title page."},
      {"field": 2, "value": "A two-stage stratified cluster sample.", "evidence": "two-stage stratified cluster sample"},
      {"field": 3, "value": "sample survey data", "evidence": "Sample survey data"},
      {"field": 1, "value": "Survey of 2031", "evidence": "invented"},
      {"field": 4, "value": "HIS", "evidence": "Household"}
    ]}
    ```"""
    cards, why = [], []
    for item in parse_proposals(reply):
        card, reason = accept(item, pointers, FILL_FIELDS, doc, doc, "12")
        (cards if card else why).append(card or reason)
    assert [c["pointer"] for c in cards] == ["/study_desc/title", "/study_desc/sampling", "/study_desc/kind"]
    assert cards[0]["status"] == "empty" and cards[1]["status"] == "improvement"
    assert cards[2]["value"] == "Sample survey data"  # the option's own spelling
    assert cards[1]["current"] == "Random sample." and cards[0]["source"] == "document"
    assert len(why) == 2  # the invented year and the list field


def test_document_fill_job_runs_off_the_fifo_queue_and_reports_progress(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "jobs").mkdir()
    reply = json.dumps({"proposals": [{"field": 1, "value": "Household Income Survey 2025", "evidence": "Household Income Survey 2025"}]})
    app = make_app(mock_response=reply)
    app.jobs = {}
    bundle = {"record_id": "12", "fill": True, "fields": FILL_FIELDS}
    with TestClient(app) as client:
        bad = client.post("/assistant/review/jobs", json={"bundle": bundle, "kind": "document"})
        assert bad.status_code == 400  # no document
        r = client.post("/assistant/review/jobs", json={"bundle": bundle, "kind": "document", "document": "Household Income Survey 2025."})
        jobid = r.json()["job_id"]
        wait_for_job(app, jobid)
    job = app.jobs[jobid]
    assert job["jobtype"] == "assistant-document-fill" and job["status"] == "done"
    assert job["info"]["progress"] == {"done": 3, "total": 3}
    stored = json.loads((tmp_path / "jobs" / f"{jobid}.json").read_text(encoding="utf-8"))
    assert [p["pointer"] for p in stored["proposals"]] == ["/study_desc/title"]
    assert "Checked 3 fields" in stored["text"]
    assert app.assistant_inflight == 0


def test_review_with_a_document_checks_every_field_the_same_way_each_time(tmp_path, monkeypatch):
    """The agent review proposed 0 to 7 fields per run on a blank record; with a document it now runs the fill pass."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "jobs").mkdir()
    reply = json.dumps({"proposals": [{"field": 1, "value": "Household Income Survey 2025", "evidence": "Household Income Survey 2025"}]})
    app = make_app(mock_response=reply)
    app.jobs = {}
    bundle = {"record_id": "12", "fill": True, "fields": FILL_FIELDS}
    runs = []
    with TestClient(app) as client:
        for _ in range(3):
            r = client.post("/assistant/review/jobs", json={"bundle": bundle, "document": "Household Income Survey 2025."})
            jobid = r.json()["job_id"]
            wait_for_job(app, jobid)
            runs.append(app.jobs[jobid])
    for job in runs:
        assert job["jobtype"] == "assistant-review" and job["status"] == "done"  # still a review to the panel
        assert job["info"]["question"] == "Review this record against the attached document"
    stored = [json.loads((tmp_path / "jobs" / f"{j['jobid']}.json").read_text(encoding="utf-8")) for j in runs]
    assert all([p["pointer"] for p in s["proposals"]] == ["/study_desc/title"] and "Checked 3 fields" in s["text"] for s in stored)


def test_review_without_fill_permission_stays_an_agent_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "jobs").mkdir()
    app = make_app(mock_response="Nothing to propose.")
    app.jobs = {}
    bundle = {"record_id": "12", "fill": False, "fields": FILL_FIELDS}
    with TestClient(app) as client:
        r = client.post("/assistant/review/jobs", json={"bundle": bundle, "document": "Household Income Survey 2025."})
        jobid = r.json()["job_id"]
        wait_for_job(app, jobid)
    stored = json.loads((tmp_path / "jobs" / f"{jobid}.json").read_text(encoding="utf-8"))
    assert "Checked" not in stored["text"]


def test_review_job_ids_are_unique_within_the_same_instant(monkeypatch):
    from src.routers import assistant as routes

    monkeypatch.setattr(routes.time, "time", lambda: 1700000000.0)
    ids = {routes._new_jobid() for _ in range(50)}
    assert len(ids) == 50 and all(i.startswith("job-1700000000.000000-") for i in ids)


def test_a_turn_waiting_for_a_slot_is_told_so():
    app = make_app()
    app.assistant_sem = asyncio.Semaphore(1)

    async def hold_then_release():
        await app.assistant_sem.acquire()
        asyncio.get_running_loop().call_later(0.3, app.assistant_sem.release)

    with TestClient(app) as client:
        client.portal.call(hold_then_release)
        r = client.post("/assistant/chat/stream", json={"question": "What does the abstract field hold?"})
    names = [n for n, _ in events(r)]
    assert names[0] == "queued" and names[-1] == "done"
    assert app.assistant_inflight == 0


def test_option_matching_accepts_the_label_or_code_without_the_bracket():
    from src.assistant.host.tools import _match_option

    options = ["Face-to-face [f2f]", "Telephone interview", "Sample survey data (ssd)"]
    assert _match_option("face-to-face", options) == "Face-to-face [f2f]"
    assert _match_option("F2F", options) == "Face-to-face [f2f]"
    assert _match_option("Sample survey data", options) == "Sample survey data (ssd)"
    assert _match_option("telephone interview", options) == "Telephone interview"
    assert _match_option("face", options) is None


# ---------------------------------------------------------------- variable labelling job, persistence, caps

class FakeStore:
    def __init__(self):
        self.saved, self.synced = [], []

    def save_queued_job(self, job):
        self.saved.append(dict(job))

    def sync_from_memory(self, job):
        self.synced.append(dict(job))


CODEBOOK = "| Variable | Label |\n|---|---|\n| hhid | Household identifier |\n| age | Age of household member (years) |\n"
VAR_BUNDLE = {
    "record_id": "12",
    "file_id": "F1",
    "datafile": {"file_id": "F1", "file_name": "hh.dta", "variables": [{"uid": 1, "name": "hhid", "label": ""}, {"uid": 2, "name": "age", "label": ""}]},
}


def test_variable_labelling_job_stores_items_and_the_job_survives_in_the_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "jobs").mkdir()
    app = make_app()
    app.jobs = {}
    app.job_store = FakeStore()
    with TestClient(app) as client:
        assert client.post("/assistant/review/jobs", json={"kind": "variables", "bundle": VAR_BUNDLE}).status_code == 400  # no document
        r = client.post("/assistant/review/jobs", json={"kind": "variables", "bundle": VAR_BUNDLE, "document": CODEBOOK, "owner": "u1"})
        jobid = r.json()["job_id"]
        wait_for_job(app, jobid)
    assert app.jobs[jobid]["jobtype"] == "assistant-variable-fill" and app.jobs[jobid]["status"] == "done"
    stored = json.loads((tmp_path / "jobs" / f"{jobid}.json").read_text(encoding="utf-8"))
    assert [(i["uid"], i["field"], i["value"]) for i in stored["variables"]["items"]] == [
        (1, "label", "Household identifier"),
        (2, "label", "Age of household member (years)"),
    ]
    # In the store from the start as processing (so a restart marks it interrupted), and again at the end.
    assert app.job_store.saved[0]["status"] == "processing"
    assert app.job_store.synced[-1]["status"] == "done"
    assert app.assistant_job_owner == {}


def test_one_person_cannot_run_more_jobs_than_the_cap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "jobs").mkdir()
    app = make_app()
    app.jobs = {}
    app.assistant_max_jobs_per_owner = 1
    app.assistant_job_owner = {"job-running": "u1"}
    body = {"kind": "variables", "bundle": VAR_BUNDLE, "document": CODEBOOK}
    with TestClient(app) as client:
        assert client.post("/assistant/review/jobs", json={**body, "owner": "u1"}).status_code == 429
        ok = client.post("/assistant/review/jobs", json={**body, "owner": "u2"})
        assert ok.status_code == 200
        wait_for_job(app, ok.json()["job_id"])


# ---------------------------------------------------------------- project knowledge sources


def _knowledge_app(tmp_path, monkeypatch, mock_response="ok"):
    from src.assistant.core.knowledge import KnowledgeStore

    monkeypatch.setenv("STORAGE_PATH", str(tmp_path))
    app = make_app(mock_response)
    app.assistant_knowledge = KnowledgeStore(tmp_path / "cache", None)
    return app


def _source(tmp_path, name="guide.md", text="# Sampling\n\nStratified random sampling of establishments.\n"):
    p = tmp_path / "docs" / name
    p.parent.mkdir(exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_sync_indexes_a_resource_and_reports_it(tmp_path, monkeypatch):
    app = _knowledge_app(tmp_path, monkeypatch)
    path = _source(tmp_path)
    with TestClient(app) as client:
        r = client.post("/assistant/knowledge/sync", json={"sid": "7", "files": [{"id": "res-1", "kind": "resource", "name": "guide.md", "path": str(path)}]})
    assert r.status_code == 200
    (source,) = r.json()["sources"]
    assert source["id"] == "res-1" and source["kind"] == "resource" and source["chunks"] >= 1 and len(source["hash"]) == 64
    assert source["state"] == "ready" and not source["truncated"]


def test_sync_refuses_paths_outside_storage_and_reports_unreadable_files(tmp_path, monkeypatch):
    app = _knowledge_app(tmp_path, monkeypatch)
    bad = _source(tmp_path, "data.bin", "x")
    with TestClient(app) as client:
        outside = client.post("/assistant/knowledge/sync", json={"sid": "7", "files": [{"id": "res-1", "path": "/etc/passwd"}]})
        unreadable = client.post("/assistant/knowledge/sync", json={"sid": "7", "files": [{"id": "res-2", "path": str(bad)}]})
    assert outside.json()["sources"][0]["state"] == "failed"
    assert unreadable.json()["sources"][0]["state"] == "failed" and "Cannot read" in unreadable.json()["sources"][0]["error"]


def test_attachments_are_registered_listed_and_removed_per_project(tmp_path, monkeypatch):
    app = _knowledge_app(tmp_path, monkeypatch)
    path = _source(tmp_path)
    with TestClient(app) as client:
        synced = client.post("/assistant/knowledge/sync", json={"sid": "7", "files": [{"id": "x", "kind": "attachment", "name": "guide.md", "path": str(path)}]}).json()["sources"][0]
        assert synced["id"].startswith("att-")
        assert [s["hash"] for s in client.post("/assistant/knowledge/list", json={"sid": "7"}).json()["sources"]] == [synced["hash"]]
        assert client.post("/assistant/knowledge/list", json={"sid": "8"}).json()["sources"] == []  # another project sees nothing
        assert client.post("/assistant/knowledge/remove", json={"sid": "8", "hashes": [synced["hash"]]}).json()["removed"] == 0
        assert client.post("/assistant/knowledge/remove", json={"sid": "7", "hashes": [synced["hash"]]}).json()["removed"] == 1
        assert client.post("/assistant/knowledge/list", json={"sid": "7"}).json()["sources"] == []


def _meta(client, body):
    meta = None
    with client.stream("POST", "/assistant/chat/stream", json=body) as r:
        assert r.status_code == 200
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event == "meta":
                import json as _json

                meta = _json.loads(line[5:])
    return meta


def test_chat_with_sources_cites_source_and_page_and_ignores_other_projects(tmp_path, monkeypatch):
    big = "# Intro\n\n" + "\n\n".join(f"## Part {n}\n\n" + "Filler text about logistics. " * 40 for n in range(40)) + "\n\n## Weights\n\nThe dataset contains two weight variables, wt_ES and wt_fu.\n"
    app = _knowledge_app(tmp_path, monkeypatch)
    path = _source(tmp_path, "methods.md", big)
    with TestClient(app) as client:
        sid_sources = [{"kind": "resource", "id": "res-1", "name": "methods.md", "path": str(path)}]
        meta = _meta(client, {"question": "What weight variables does the dataset contain?", "sid": "7", "sources": sid_sources, "mode": "app"})
        assert "document:excerpt" in meta["sources"] and any(c.startswith("methods.md") for c in meta["citations"])
        # An attachment hash that was never registered under this project is skipped, never read.
        entry = app.assistant_knowledge.get(next(iter(app.assistant_knowledge._entries)))
        stray = [{"kind": "attachment", "id": "att-1", "hash": entry.hash}]
        r = client.post("/assistant/chat/stream", json={"question": "weights?", "sid": "9", "sources": stray, "mode": "app"})
        assert r.status_code == 400 and "None of the selected sources" in r.json()["detail"]


# ---- a project that must stay on this server ---------------------------------------------------

def test_keep_local_never_reaches_a_hosted_model():
    from src.routers.assistant import _local_spec

    registry = make_app().assistant_gateway.registry
    assert _local_spec(registry, "hosted").alias == "local"  # a hosted choice is replaced, not honoured
    assert _local_spec(registry, None).alias == "local"
    assert _local_spec(registry, "no-such-model").alias == "local"


def test_keep_local_without_a_local_model_is_refused():
    from fastapi import HTTPException
    from src.routers.assistant import _local_spec

    reg = parse_registry({"models": [{"alias": "hosted", "local": False, "litellm_params": {"model": "openai/x", "api_key": "k"}}]}, {})
    with pytest.raises(HTTPException) as err:
        _local_spec(reg, None)
    assert err.value.status_code == 409


def test_a_local_model_cannot_be_configured_to_fall_back_to_a_hosted_one():
    from src.assistant.core.config import ConfigError

    models = [
        {"alias": "l", "local": True, "litellm_params": {"model": "openai/t", "api_base": "http://127.0.0.1:9/v1", "api_key": "none"}},
        {"alias": "h", "local": False, "litellm_params": {"model": "openai/x", "api_key": "k"}},
    ]
    with pytest.raises(ConfigError):
        parse_registry({"models": models, "fallbacks": {"l": ["h"]}}, {})
    parse_registry({"models": models, "fallbacks": {"h": ["l"]}}, {})  # hosted to local is fine


def test_chat_with_keep_local_is_answered_by_the_local_model_even_if_hosted_is_asked_for():
    app = make_app()
    r = post(app, model="hosted", keep_local=True)
    assert r.status_code == 200
    done = [d for n, d in events(r) if n == "done"][0]
    assert done.get("model") in (None, "local") and "hosted" not in json.dumps(done)


def test_model_error_does_not_list_the_router_fallback_table():
    # LiteLLM's router appends every configured fallback (hosted models included) to the provider error.
    from src.assistant.core.chat import _error_message

    exc = RuntimeError(
        "litellm.InternalServerError: OpenAIException - Connection error.No fallback model group found for "
        "original model_group=local-lfm. Fallbacks=[{'openrouter-gpt6': ['openrouter-sonnet', 'local-lfm']}]. "
        "Received Model Group=local-lfm"
    )
    message = _error_message(exc)
    assert "openrouter" not in message
    assert "Fallbacks" not in message
    assert message == "RuntimeError: litellm.InternalServerError: OpenAIException - Connection error"
