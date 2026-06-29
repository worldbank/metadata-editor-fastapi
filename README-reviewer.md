# Metadata Reviewer – Installation Guide

The metadata reviewer is an **optional** feature that uses LLM agents (via the [ai4data](https://pypi.org/project/ai4data/) package) to scan metadata documents for quality issues — inconsistencies, typos, missing information, and more.

It is not included in the base `requirements.txt`. Install it separately when you need `/review/*` endpoints.

---

## Prerequisites

- Python 3.11 or later
- Access to an LLM provider (Azure OpenAI, OpenAI, Ollama, or Anthropic)

---

## Step 1 – Install base dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 – Install reviewer packages

The default `requirements-reviewer.txt` targets **Azure OpenAI**:

```bash
pip install -r requirements-reviewer.txt
```

Or with uv (quote brackets in zsh):

```bash
uv pip install -r requirements-reviewer.txt
```

### Other providers

Edit `requirements-reviewer.txt` or install directly:

```bash
# OpenAI
pip install 'ai4data[metadata-reviewer,openai]>=0.1.0'

# Ollama (local)
pip install 'ai4data[metadata-reviewer,ollama]>=0.1.0'

# Anthropic
pip install 'ai4data[metadata-reviewer,anthropic]>=0.1.0'
```

---

## Step 3 – Configure credentials

Copy the example env file next to `main.py`:

```bash
cp reviewer.env.example reviewer.env
```

Edit `reviewer.env` for your provider. Key variables:

| Variable | Description |
|----------|-------------|
| `REVIEWER_PROVIDER` | `azure`, `openai`, `ollama`, or `anthropic` |
| `REVIEWER_MODEL` | Model or deployment name |
| `REVIEWER_CONCURRENCY` | Max simultaneous reviews (default: 10) |
| `REVIEWER_MAX_INFLIGHT` | Max tracked jobs (default: 200) |
| `REVIEWER_JOB_TIMEOUT_SEC` | Per-job timeout in seconds (default: 900) |

See `reviewer.env.example` for Azure API key vs Entra ID client-credentials setup.

`reviewer.env` is gitignored. Legacy unprefixed names (`AZURE_OPENAI_*`, `OPENAI_API_KEY`, etc.) in `.env` still work as fallbacks.

---

## Step 4 – Start the service

```bash
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
```

On startup you should see either:

- `Metadata reviewer routes registered at /review` — package installed and configured
- `Metadata reviewer not installed (optional)` — base app runs without reviewer endpoints

---

## API usage

### List available agent manifests

```bash
curl http://localhost:8000/review/manifests
```

### Submit a review job

```bash
curl -X POST http://localhost:8000/review/jobs \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"title": "My dataset", "description": "..."}}'
```

Returns `202` with a `job_id`.

### Poll for results

Use the shared jobs API (same as CSV and data-dictionary jobs):

```bash
curl http://localhost:8000/jobs/job-1234567890.123
```

When `status` is `done`, the response includes a `data` field with the JSON array of detected issues.

### Cancel a job

```bash
curl -X DELETE http://localhost:8000/jobs/job-1234567890.123
```

---

## Without the reviewer installed

The core FastAPI service starts normally. `/review/*` routes are not registered. No reviewer-specific configuration is required.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| No `/review` routes in `/docs` | Run `pip install -r requirements-reviewer.txt` |
| `503` on `POST /review/jobs` | Missing or invalid credentials in `reviewer.env` |
| `429 Too many in-flight reviewer jobs` | Lower load or raise `REVIEWER_MAX_INFLIGHT` |
| Job `error` with timeout message | Increase `REVIEWER_JOB_TIMEOUT_SEC` or use a faster model |

---

## Further reading

- [ai4data documentation](https://worldbank.github.io/ai4data)
- [ai4data on PyPI](https://pypi.org/project/ai4data/)
