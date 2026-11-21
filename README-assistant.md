# Assistant (server-side chat)

Optional. Serves the Metadata Editor's assistant: a chat panel that answers questions about the editor
and the open project, drafts field values from the project's documents and reviews a record. The
editor's PHP proxy (`api/assistant/*`, see `README-assistant.md` in the editor) checks the session and
project permissions and calls the routes below; the browser never reaches this service.

Every model call goes through [LiteLLM](https://docs.litellm.ai/) using OpenAI-compatible chat
completions, so local models (Ollama, llama.cpp `llama-server`, LM Studio, vLLM) and hosted ones
(OpenAI, Azure OpenAI, Anthropic, Gemini, OpenRouter) are interchangeable by configuration.
Retrieval (BM25 plus a local embedding model) always runs on this server.

Nothing changes for an install that does not use it: without `requirements-assistant.txt` installed the
`/assistant` routes are not mounted, exactly like the metadata reviewer.

## Install

The service runs as installed by `deploy/linux/install-service.sh` (systemd) or
`deploy/windows/install-service.bat` (NSSM), on the same server as the editor. Nothing else changes in
that setup: one uvicorn process, bound to 127.0.0.1.

1. In the service's conda environment:

   ```sh
   conda activate metadata-editor
   pip install -r requirements-assistant.txt
   ```

   `requirements-assistant.txt` pins LiteLLM to the tested release (1.106.1 failed every tool call with
   `No module named 'mcp'`).

2. Next to `main.py`:

   ```sh
   cp assistant.env.example assistant.env
   cp assistant.models.yaml.example assistant.models.yaml
   ```

   Edit `assistant.models.yaml` to list the models you serve (see [Models](#models)) and put provider
   keys in `assistant.env`. Point `ASSISTANT_HELP_PATH` and `ASSISTANT_DOCS` at the editor's folder (see
   the top of the example). Both files are git-ignored.

   `assistant.env` and `.env` hold secrets, so let only an administrator and the service read them.
   Linux: `sudo chown root:editor-fastapi assistant.env .env && sudo chmod 640 assistant.env .env` (the
   service fails at start if it cannot read `.env`). Windows: limit both files to Administrators and the
   account the service runs as.

3. Optional, the documentation the assistant searches: run `node tools/assistant-docs.mjs` in the editor
   (Node 18+; it mirrors the published editor and schema docs into `assistant-docs/`). The folder is
   generated and git-ignored, so a checkout has none until it runs. Without it the assistant still
   works; `GET /assistant/status` shows retrieval `failed` and chat answers without the docs.

4. Behind a proxy that re-signs TLS, set `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` to a bundle that holds
   its CA (see the example). Python does not use the system's CA list, so hosted models and the
   one-time embedding download fail with `CERTIFICATE_VERIFY_FAILED` even when `curl` works.

5. The service writes `db/`: the embedding model (~605 MB, downloaded on first start), the Hugging Face
   client cache (`db/hf`), the retrieval index and the sources cache. `install-service.sh` creates `db/`
   and hands it to the service user; on an existing install run `sudo chown -R editor-fastapi: db`.
   `ASSISTANT_MODEL_CACHE` moves the model (`HF_HOME` then defaults to `hf/` next to it).

   The service also reads files the editor writes: project resources and chat attachments
   (`datafiles/assistant-tmp`, created group-readable). It reads them through the shared group that the
   install scripts set up when `STORAGE_PATH` and `WEB_SERVER_USER` are given; without it, attachments
   fail.

6. Restart the service (`sudo systemctl restart editor-fastapi`, or `nssm restart editor-fastapi`) and
   check:

   ```sh
   curl -s localhost:8000/assistant/status
   ```

7. In the editor, set `server_assistant_enabled` to `true` in `application/config/editor.php`.
   `data_api_url` there must reach this service.

Routes are mounted at `/assistant` when LiteLLM is installed and `ASSISTANT_ENABLED` is not `0`. A
missing or invalid model registry keeps the routes mounted; `GET /assistant/status` reports the problem
and chat returns 503.

### Web server

The editor relays the chat stream as `text/event-stream`. Apache with mod_php streams as is. With
PHP-FPM behind Apache, add `flushpackets=on` to the `ProxyPass` / `SetHandler "proxy:fcgi://..."` line;
behind Nginx the proxy sends `X-Accel-Buffering: no`, which turns `fastcgi_buffering` off. On IIS, set
`responseBufferLimit="0"` on the PHP FastCGI handler. Without that, answers arrive in one piece at the
end instead of as they are written.

A chat turn or draft can run for minutes. The proxy lifts PHP's time limit itself, but a web server
timeout below 300 s cuts it: Apache `Timeout`, Nginx `fastcgi_read_timeout`, IIS FastCGI
`activityTimeout` and `requestTimeout`.

## Models

The model is whatever `default:` names in `assistant.models.yaml`. The example ships a local model
(LFM2.5-2.6B on llama.cpp) as the default and lists hosted ones that stay unconfigured until their key
is set. Choose the provider your organisation has approved for project metadata: record text, field
values and passages of the project's documents are sent to the model.

- `local: true` marks a model that runs on your own infrastructure. `ASSISTANT_LOCAL_ONLY=1` hides
  every other model.
- `fallbacks:` names models to try when one fails. A project an administrator marks "Keep on this
  server" in the editor only uses local models and never falls back to a hosted one.
- A model whose key is unset is listed as unconfigured and never called. A rejected key (401) or an
  exhausted budget (403) surfaces at the first call; the panel shows the provider's message under
  Details.

Any OpenAI-compatible local server works:

```sh
# llama.cpp; --jinja enables the model's chat template and tool calling
llama-server -hf LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M --jinja -c 32768 --port 8080

# Ollama
ollama pull qwen3:8b
```

## Document retrieval

With `ASSISTANT_DOCS` set, each turn is grounded in the most relevant sections of those documents:
BM25 plus dense embeddings, fused 50/50, top 5 sections under 6 KB. Embeddings are local and free:

- **UltraBERT-Embed-150M** (dmis-lab, Apache-2.0, 768 dims) on ONNX Runtime, CPU, no PyTorch. The ONNX
  file is pinned by revision and SHA-256 and was checked against the official weights.
- Downloaded once (~605 MB) into `ASSISTANT_MODEL_CACHE`. The index is saved to `ASSISTANT_INDEX_PATH`
  and rebuilt only when the documents, chunking or embedder change (~2 min on CPU for the editor docs).
- Keyword search answers from the first request while embeddings build; `GET /assistant/status` shows
  the state (`keyword`, `embedding`, `ready`, `failed`).
- `ASSISTANT_EMBEDDER=none` keeps keyword search only.

## Project sources (knowledge index)

Files a curator works from (External resources, chat attachments) are read in full (page by page),
chunked, indexed with BM25 and the embedder, and cached by SHA-256 under `ASSISTANT_KNOWLEDGE_DIR`.
Nothing is sent to a model whole: a turn or fill retrieves labelled passages (`name p.N`), one query per
field. Chat, suggest and review requests take `sid` and `sources` and search only what that project
registered. `ASSISTANT_KNOWLEDGE_MAX_MB` caps the disk cache.

A fill from a document runs in batches of six fields, one completion each, and keeps a proposal only
when the document carries it (quoted evidence or the value itself) and it passes the same guard as
`propose_field_value`.

List fields (`host/listfill.py`): the proxy marks a plain `array` field with its columns, and the
document fill adds a second pass asking for rows to add. A row is kept only when its first column is
written in the document; saved rows are never touched. Columns with a controlled list are left out.

Labelling variables (`host/varfill.py`) reads the codebook first and the model second. Table rows and
`name: label` / `name - label` runs are taken directly, so a codebook of thousands of variables costs no
model calls. Only variables no line states go to the model, a dozen per completion, up to 240 per run.
A model reading is kept only when its evidence is in the document and names the variable. Saved values
are never replaced by the model; a codebook line that disagrees with a saved label is offered as
"differs".

## Tools and the agent loop

A turn on a record page can call tools: at most 4 calls per round, 5 per turn, 5 rounds plus an
answer-only round, argument validation against each tool's schema, repeat calls answered from cache,
10 s per call, result byte caps, and one repair round when the reply is empty, plan-shaped or cut off.

| Tool | When offered | Reads |
|---|---|---|
| `get_datetime`, `calculate` | every turn except greetings | nothing (offline) |
| `list_fields` | record page | field pointer, label, section, empty / required / recommended; `filter: fillable` gives what a proposal may target |
| `read_fields` | record page | saved values of up to 15 fields per call, with help text and controlled-list options |
| `get_current_page` | record page | the open data file and its variables, else the record |
| `list_data_files` | record page | the project's data files: name, cases, variable count, description |
| `find_variables` | a data file is open | BM25 plus embedding search over its variables, or the ones missing a label or description |
| `search_docs` | record page | the editor docs, DDI schema guide and API reference |
| `search_document` | sources ticked (or an attached document) | passages of the project's sources, labelled `name p.N` |
| `propose_field_value` | record page, the curator may edit | emits a `proposal` event; never writes |
| `propose_datafile_description` | the curator may edit, a data file open | emits a `proposal` event; never writes |

Tools read only what the editor's proxy put in the request; this service never calls the editor and
never writes a record. A proposal reaches a field only when the curator clicks "Fill in form", and the
editor then treats it like typing: nothing is saved until the curator saves.

How calls reach the model is per model (`tool_mode` in the registry):

- `native`: OpenAI `tools` / `tool_calls`. For llama-server, start it with `--jinja`.
- `text`: the protocol in `text_protocol` (`lfm` or `hermes`) is taught in the system prompt and parsed
  from the reply; call markers are hidden from the token stream.
- unset: LiteLLM's capability table decides; unknown (most local) models get the text protocol.

## Routes

These are internal: the editor's PHP proxy calls them after its own session and ACL checks. Never expose
this port.

| Method | Path | |
|---|---|---|
| GET | `/assistant/status` | enabled, default model, retrieval state, config error if any |
| GET | `/assistant/models` | usable aliases; `available` probes local servers (cached 30 s) |
| POST | `/assistant/chat/stream` | one turn, `text/event-stream` |
| POST | `/assistant/suggest` | one-shot task (`datafile-description`, `resource-field`), JSON `{suggestion, reason}` |
| POST | `/assistant/review/jobs` | a job: whole-record review (`kind: "record"`), every fillable field checked against a document (`"document"`), or a data file's variables labelled from a codebook (`"variables"`); progress in `info.progress`; poll `GET /jobs/{id}`, cancel `DELETE /jobs/{id}` |
| POST | `/assistant/documents/extract` | text of a PDF/text attachment under `STORAGE_PATH` |
| POST | `/assistant/knowledge/sync`, `/list`, `/remove` | index a project's sources by path under `STORAGE_PATH`, list them, remove one |

```sh
curl -N localhost:8000/assistant/chat/stream -H 'content-type: application/json' \
  -d '{"question": "How do I create a project?"}'
```

Stream events (protocol `pslm-server/1`): `queued` when every model slot is busy, then `meta` (model,
mode, tools offered); then `reasoning`, `token`, `tool` (stages: call, result, failed, repeat, refused,
capped, truncated), `proposal`, `round` and `warning`; then exactly one `done` (final text, tool-call
count, completeness verdict, usage, timings) or `error`. `done.text` is the authoritative final answer.

## Operations

- **Logs:** one `assistant turn ...` line per chat turn (model, mode, tool calls, rounds, finish
  reason, tokens, ms), one `assistant job ...` line per review or fill. No question, record value or
  answer text is logged.
- **Jobs:** a review or fill is registered in the job store when it starts and again when it ends, so
  its result survives a restart; one still running at a restart is marked interrupted and the panel
  asks the curator to start it again. `ASSISTANT_MAX_JOBS_PER_USER` (default 2) caps one person's
  running jobs.
- **Limits:** `ASSISTANT_CONCURRENCY` bounds simultaneous model work; `ASSISTANT_MAX_INFLIGHT` bounds
  open requests and running jobs together. A turn that finds every slot busy says so (`queued`) and
  waits. Reviews and fills run as their own tasks, not on the service's FIFO job queue, so they never
  hold up data jobs. A client that disconnects stops generation within half a second.
- **Scaling:** jobs, slots and the knowledge index live in this one process, like the data job queue.
  Scale up (more slots, a faster model), not out.

All settings: `assistant.env.example`.

## Tests

```sh
pip install pytest
python -m pytest src/assistant/tests
```

No network and no model server needed; LiteLLM's `mock_response` stands in for the model.
