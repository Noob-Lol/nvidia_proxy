import os

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

app = FastAPI(title="OpenAI → NVIDIA Proxy")

# NVIDIA API base (you can override with env if needed)
NVIDIA_BASE = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1")


@app.get("/health")
async def health():
    """Simple health check."""
    return PlainTextResponse("ok")


@app.get("/v1")
async def root_v1():
    """Metadata endpoint for compatibility."""
    return JSONResponse({"object": "list", "endpoints": ["/v1/models", "/v1/chat/completions"]})


@app.get("/v1/models")
async def models(request: Request):
    """List available models (passthrough to NVIDIA)."""
    headers = _forward_headers(request)
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{NVIDIA_BASE}/models", headers=headers)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy(request: Request, path: str):
    """Proxy any /v1/* endpoint transparently to NVIDIA API."""
    url = f"{NVIDIA_BASE}/{path}"
    headers = _forward_headers(request)
    body = await request.body()

    # Detect streaming request (client sets "stream": true)
    stream = False
    if request.method == "POST":
        try:
            data = await request.json()
            stream = bool(data.get("stream", False))
        except Exception:
            pass

    async with httpx.AsyncClient(timeout=None) as client:
        if stream:
            # Stream response (Server-Sent Events passthrough)
            async with client.stream(request.method, url, headers=headers, content=body) as r:
                return StreamingResponse(
                    _iter_stream(r),
                    media_type=r.headers.get("content-type", "text/event-stream"),
                    status_code=r.status_code,
                )
        else:
            # Regular JSON response
            r = await client.request(request.method, url, headers=headers, content=body)
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type"),
            )


def _forward_headers(request: Request) -> dict:
    """Forward headers, ensuring BYOK (user-provided Authorization)."""
    headers = {
        "Content-Type": request.headers.get("content-type", "application/json"),
    }
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


async def _iter_stream(response: httpx.Response):
    """Stream event chunks from NVIDIA → client."""
    async for chunk in response.aiter_bytes():
        yield chunk


# Optional root
@app.get("/")
async def index():
    return PlainTextResponse("OpenAI-compatible NVIDIA proxy is running.")
