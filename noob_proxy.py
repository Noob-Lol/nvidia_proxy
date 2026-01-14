import json
import os
from functools import partial

from aiohttp import ClientSession, TCPConnector, web

NVIDIA_BASE = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1")
PORT = int(os.getenv("PORT", 8007))


# We'll attach the session to the app object
async def create_app():
    app = web.Application(middlewares=[cors_middleware])

    # Create global session on startup
    async def on_startup(app):
        app["session"] = ClientSession(connector=TCPConnector(ttl_dns_cache=300))
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

    async def route_list(_):
        return web.json_response({"object": "list", "endpoints": ["/v1/models", "/v1/chat/completions"]})

    partial_options_get = partial(options_handler, methods="GET")
    get_routes_dict = {"/": root, "/health": health, "/v1": route_list, "/v1/models": proxy}
    for route, handler in get_routes_dict.items():
        app.router.add_get(route, handler)
        app.router.add_options(route, partial_options_get)
    partial_options_post = partial(options_handler, methods="POST")
    app.router.add_post("/v1/chat/completions", proxy)
    app.router.add_options("/v1/chat/completions", partial_options_post)
    # we don't need more routes.
    return app


@web.middleware
async def cors_middleware(request: web.Request, handler):
    response = await handler(request)

    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Vary"] = "Origin"

    return response


async def options_handler(request: web.Request, methods=""):
    """CORS preflight handler"""
    # idk if options is needed, but I will add it
    methods += ", OPTIONS" if methods else "OPTIONS"
    origin = request.headers.get("Origin")
    req_method = request.headers.get("Access-Control-Request-Method")

    # Not a CORS preflight → let aiohttp routing decide
    if not origin or not req_method:
        raise web.HTTPMethodNotAllowed(
            method="OPTIONS",
            allowed_methods=methods.split(", "),
        )

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": methods,
        "Vary": "Origin",
    }

    # Echo requested headers if present
    req_headers = request.headers.get("Access-Control-Request-Headers")
    if req_headers:
        headers["Access-Control-Allow-Headers"] = req_headers

    return web.Response(status=204, headers=headers)


async def proxy(request: web.Request):
    """Transparent proxy handler"""
    path = request.match_info.get("path", "")
    url = f"{NVIDIA_BASE}/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    body = await request.read()
    stream = False
    if request.method == "POST":
        try:
            stream = json.loads(body).get("stream", False)
        except Exception:
            stream = False
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
            return web.Response(body=data, status=resp.status, content_type=resp.content_type)


if __name__ == "__main__":
    web.run_app(create_app(), host="0.0.0.0", port=PORT)
