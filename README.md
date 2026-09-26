# FinDocs Compliance Intelligence Engine

Enterprise financial-document ingestion, deterministic quantitative analysis, and compliance intelligence.

## Phase 1: run the platform

1. Copy `.env.example` to `.env` and set environment-specific values.
2. Start the stack: `docker compose up --build`.
3. Verify the API at `http://localhost:8000/health` and documentation at `http://localhost:8000/docs`.

Local service endpoints: API `:8000`, dashboard `:8501`, Qdrant `:6333`, Prometheus `:9090`, Grafana `:3000`.

Use `/health` for process liveness and `/ready` for dependency readiness. Set `APP_API_KEY` in production; all `/api/v1/*` routes then require the `X-API-Key` header. Production deployments must inject `.env` values through a secret manager; never commit `.env` or use example Grafana credentials.

Prometheus scrapes `GET /api/v1/metrics`. The service reports liveness, dependency readiness, request metrics, document-ingestion outcomes, agent workflows, and durable audit logging.

## Phase 2: document ingestion

`POST /api/v1/upload` accepts multipart form field `document` for `.pdf` and `.csv` sources. Files are streamed to the configured storage directory, capped at `MAX_UPLOAD_BYTES`, and processed after a `202 Accepted` response.

- CSV data is converted locally into Markdown tables, preserving every source row and column.
- PDFs are parsed by LlamaParse in Markdown mode, preserving headings and detected tables.
- The chunker keeps legal narrative at paragraph/sentence boundaries and never splits a table row; each table chunk repeats its header.
- Gemini's `gemini-embedding-001` produces retrieval-document vectors, which are stored in Qdrant with document, section, page, and chunk citation metadata.

Set both `GEMINI_API_KEY` and `LLAMA_CLOUD_API_KEY` before ingesting production documents. The API accepts uploads without credentials, but records background processing as failed rather than fabricating an ingestion result. Document ingestion status is persisted in the configured SQL database, so status survives API restarts.

## Phase 3: agent analysis

`POST /api/v1/query` accepts `{"doc_id": "...", "query": "..."}` and invokes a compiled LangGraph state machine. The supervisor selects a retrieval, compliance, quantitative, or combined route; retrieval is always scoped to `doc_id`.

Compliance findings are derived from JSON rules in `app/rules/credit_agreement_rules.json`. The quantitative agent supports debt-to-equity, current ratio, debt-to-EBITDA, and interest coverage. It extracts only exact table values from cited passages and executes a fixed, AST-validated `Decimal` template in an isolated Python process. It returns an explicit unavailable/failed result if operands are missing or execution fails; it never estimates a financial number.

## Phase 4: auditability and resilience

Every `POST /api/v1/query` receives an `X-Request-ID`. A row in `query_audit_logs` records the document ID, query, supervisor route, completed agent steps, Gemini token usage, latency, structured final response, and safe error category. Development defaults to SQLite; set `DATABASE_URL=postgresql+psycopg://...` for PostgreSQL.

Prometheus now exposes document-analysis request outcomes, LangGraph duration, LLM token counts, and restricted-code outcomes. The API container runs with a read-only root filesystem, dropped Linux capabilities, no-new-privileges, a constrained PID count, and a small temporary filesystem. `CODE_EXECUTION_TIMEOUT_SECONDS` limits the local calculation worker; a production deployment should place that worker in a dedicated jailed workload or microVM.

## Production requirements

- Use PostgreSQL and apply schema changes through a migration tool before deploying.
- Rotate exposed credentials and inject replacement values through the deployment secret manager.
- Put the calculation subprocess in a dedicated sandbox or microVM with a separate service account.
- Add authentication, authorization, rate limiting, TLS termination, backup/restore, and retention policies at the deployment boundary.

The production Compose overlay provides PostgreSQL, Alembic startup migrations, Caddy HTTPS termination, and daily seven-day rotating database dumps:

```powershell
docker compose -f docker-compose.yml -f docker-compose.production.yml --env-file .env.production up -d --build
```

Copy `.env.production.example` to `.env.production`, populate it from a secret manager, and configure DNS for `PUBLIC_HOST`. Caddy obtains certificates automatically. The calculation runner remains an isolated, no-network Python subprocess inside the API container; a regulated deployment must replace it with a separately jailed worker or microVM before handling untrusted workloads.

## Local test

Install dependencies with `pip install -r requirements.txt`, then run `pytest -q`.
