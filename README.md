# agent-container

A real-estate research agent, extracted from the Deal Agent's `rag/` service: LangGraph + LangChain, with every model call going through [OpenRouter](https://openrouter.ai). No website or UI, only an HTTP API. It runs as a Docker container or as a Vercel Function.

```
start_turn ─▶ generate_query_or_respond ──(tool call)──▶ retrieve ──▶ grade ──▶ generate_answer
                        ▲                                               │
                        └────────────── rewrite_question ◀──(irrelevant)┘   (max 2 rewrites per turn)
```

Adapted from the [LangGraph agentic RAG tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag). The agent has two tools:

- `search_properties`: semantic search over the listings in `data/raw/`, with hard filters (city, ZIP, price, beds, type, for sale).
- `get_property_data`: live subject facts and comparables for one street address via an Apify Zillow Actor (needs `APIFY_TOKEN`), cached in SQLite for 24 hours.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/chat` | bearer | `{"message": "..."}` → `{"answer": "..."}`, one stateless turn |
| `POST` | `/agent` | bearer | [AG-UI](https://docs.ag-ui.com) stream, for CopilotKit or any AG-UI client |
| `POST` | `/tools/property` | bearer | `{"address": "...", "refresh": false}` → the raw property-data payload |
| `GET` | `/health`, `/healthz`, `/agent/health` | open | Health checks |

"Bearer" means `Authorization: Bearer $PI_AGENT_TOKEN`, enforced whenever `PI_AGENT_TOKEN` is set. Always set it when deployed, because every call spends OpenRouter (and possibly Apify) credit.

## Configuration

Copy `.env.example` to `.env`. Unset or blank values fall back to the defaults in `src/property_intelligence_rag/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | none | **Required.** Chat and embeddings |
| `PI_CHAT_MODEL` | `anthropic/claude-opus-5.5` | Any OpenRouter model with tool calling |
| `PI_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | OpenRouter embedding model |
| `PI_AGENT_TOKEN` | none | Bearer token for the paid endpoints |
| `PI_API_SECRET` | none | Optional extra `x-pi-secret` header required on `/agent` |
| `PI_WEB_ORIGINS` | none | Comma-separated browser origins allowed by CORS (localhost is always allowed) |
| `APIFY_TOKEN` | none | Enables live address lookups |

## Run locally

```bash
uv sync
cp .env.example .env            # add OPENROUTER_API_KEY
uv run pi-ingest                # optional: embed data/raw/ now instead of on the first search
uv run pi-ask "3 bedroom homes in San Jose under 1.2m"
uv run pi-serve                 # http://127.0.0.1:8123
```

## Run as a container

```bash
docker build -t agent-container .
docker run --rm -p 8080:8080 --env-file .env agent-container

curl -X POST localhost:8080/chat \
  -H "authorization: Bearer $PI_AGENT_TOKEN" -H 'content-type: application/json' \
  -d '{"message":"Which listings have at least 3 bedrooms?"}'
```

The image builds the listings index at boot. Mount a volume at `/data` to keep the property cache across restarts.

## Deploy to Vercel

Vercel's FastAPI preset picks up `app` from the root `app.py`. No build settings are needed.

1. Import this repo in Vercel (or run `vercel` in the repo root). The framework is detected as FastAPI.
2. Under Settings → Environment Variables, add `OPENROUTER_API_KEY` and `PI_AGENT_TOKEN` (generate one with `openssl rand -hex 32`), plus any optional variables from the table above.
3. Deploy, then check `https://<project>.vercel.app/health`.

How it behaves on Vercel:

- **Writable files go to `/tmp`.** When the `VERCEL` variable is set, the index and the SQLite cache live in `/tmp/agent-container/`. Each new instance re-embeds `data/raw/` on its first search, which takes a few seconds for the sample data.
- **`/chat` is the safe endpoint.** `/agent` keeps AG-UI threads in process memory, so a follow-up that lands on a different instance loses earlier turns. For multi-turn chat on Vercel, either resend the history from the client or replace the `InMemorySaver` in `server.py` with a persistent LangGraph checkpointer (Postgres, Redis).
- **Up to 300 s per request** (`vercel.json`), enough for a cold Apify lookup (`PI_APIFY_TIMEOUT_SECONDS`, default 120).

## Data

`data/raw/sample_listings.json` holds seven fictional listings. Replace it with scraped data (`.json`, `.jsonl` or `.csv`; field names are mapped by `FIELD_ALIASES` in `schema.py`). Listings are deduplicated by id, and the latest file wins. Re-run `pi-ingest`, or redeploy, after changing the data or the embedding model.

## Layout

| File | Purpose |
|---|---|
| `app.py` | Vercel entrypoint; re-exports the FastAPI app from `server.py` |
| `Dockerfile` | Container image (`pi-serve` on port 8080) |
| `src/property_intelligence_rag/graph.py` | The LangGraph agent |
| `src/property_intelligence_rag/server.py` | FastAPI app: AG-UI, chat, property tool, health, auth, CORS |
| `src/property_intelligence_rag/retriever.py` | The agent's two tools |
| `src/property_intelligence_rag/ingest.py` | Embeds `data/raw/` into `data/index/properties.json` |
| `src/property_intelligence_rag/property_data.py` | Live comparables and market summary |
| `src/property_intelligence_rag/property_cache.py` | SQLite cache |
| `src/property_intelligence_rag/apify.py` | Minimal Apify REST client |
| `src/property_intelligence_rag/schema.py` | Maps scraper fields onto a canonical `Property` |
| `src/property_intelligence_rag/cli.py` | `pi-ask` |

## Tests

```bash
uv run pytest       # fake embeddings, no network
```
