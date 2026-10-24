# Container image for the agent: AG-UI endpoint, JSON chat and property-research tool.
#   docker build -t agent-container .
#   docker run --rm -p 8080:8080 --env-file .env agent-container
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /srv

# Dependencies first for layer caching, then the app.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY data/raw ./data/raw
COPY data/demo_property.json ./data/demo_property.json
RUN uv sync --frozen --no-dev

ENV PATH="/srv/.venv/bin:$PATH" \
    PI_PROPERTY_CACHE_PATH=/data/property-cache.sqlite3 \
    PI_AGENT_HOST=0.0.0.0 \
    PI_AGENT_PORT=8080
EXPOSE 8080

# The listings index is gitignored; build it at boot (needs OPENROUTER_API_KEY). If this fails,
# the first listing search builds it instead; the other endpoints serve either way.
CMD ["sh", "-c", "[ -f data/index/properties.json ] || pi-ingest || echo 'pi-ingest failed; index builds on first search'; exec pi-serve"]
