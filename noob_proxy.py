import json
import os

from aiohttp import ClientSession, TCPConnector, web

NVIDIA_BASE = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1")


# We'll attach the session to the app object
async def create_app():
    app = web.Application()

    # Create global session on startup
    async def on_startup(app):
        app["session"] = ClientSession(connector=TCPConnector(ttl_dns_cache=300, enable_cleanup_closed=True))
        print("ClientSession created")

    # Close it on shutdown
    async def on_cleanup(app):
        await app["session"].close()
        print("ClientSession closed")

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # Routes
    async def root(_):
        return web.Response(text="OpenAI-compatible NVIDIA proxy (aiohttp) running.")

    async def health(_):
        return web.Response(text="ok")

    async def routes(_):
        return web.json_response({"object": "list", "endpoints": ["/v1/models", "/v1/chat/completions"]})

    app.router.add_get("/", root)
    app.router.add_get("/health", health)
    app.router.add_get("/v1", routes)
    app.router.add_route("*", "/v1/{path:.*}", proxy)

    return app


async def proxy(request: web.Request):
    """Transparent proxy handler"""
    path = request.match_info.get("path", "")
    url = f"{NVIDIA_BASE}/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    body = await request.read()
    # Handle preflight CORS
    if request.method == "OPTIONS":
        return web.Response(headers={
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
            "Access-Control-Allow-Origin": "*",
        })

    stream = False
    if request.method == "POST":
        try:
            stream = json.loads(body).get("stream", False)
        except Exception:
            pass

    session: ClientSession = request.app["session"]

    if stream:
        async with session.post(url, data=body, headers=headers) as resp:
            if resp.status >= 400:
                text = await resp.text()
                return web.Response(text=text, status=resp.status, content_type=resp.content_type)

            response = web.StreamResponse(status=resp.status, headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            async for chunk, _ in resp.content.iter_chunks():
                if chunk:
                    await response.write(chunk)
            await response.write_eof()
            return response
    else:
        async with session.request(request.method, url, data=body, headers=headers) as resp:
            data = await resp.read()
            cors_headers = {"Access-Control-Allow-Credentials": "true", "Access-Control-Allow-Origin": "*"}
            return web.Response(body=data, status=resp.status, content_type=resp.content_type, headers=cors_headers)


if __name__ == "__main__":
    web.run_app(create_app(), host="0.0.0.0", port=8007)
