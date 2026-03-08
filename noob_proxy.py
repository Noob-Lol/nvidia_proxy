import logging
import os
from urllib.parse import urlparse

from aiohttp import ClientConnectorError, ClientResponse, ClientSession, ClientTimeout, InvalidURL, TCPConnector, web

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",  # Removes timestamps and logger names
)
logger = logging.getLogger(__name__)
# Headers that should not be forwarded (Hop-by-hop)
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "content-length",
}
PORT = int(os.getenv("PORT", 8007))


async def process_response(response: ClientResponse, request: web.Request) -> web.StreamResponse:
    proxy_resp = web.StreamResponse(status=response.status, reason=response.reason)
    # Copy relevant headers from target, then add CORS
    for k, v in response.headers.items():
        if k.lower() not in HOP_BY_HOP:
            proxy_resp.headers[k] = v

    proxy_resp.headers["Access-Control-Allow-Origin"] = "*"
    await proxy_resp.prepare(request)

    # 5. Defensive Streaming (The "Closed Transport" Fix)
    try:
        async for chunk in response.content.iter_any():
            await proxy_resp.write(chunk)
    except (ConnectionResetError, BrokenPipeError):
        # This happens if the user closes the tab or cancels the request
        logger.info("Client disconnected prematurely.")
    except Exception:
        logger.exception("Error during stream:")
    else:
        # write EOF only if no exceptions
        await proxy_resp.write_eof()
    return proxy_resp


async def proxy_handler(request: web.Request):
    # 1. Extract the target URL from the path
    # Usage: http://localhost:8080/https://api.example.com/data
    target_url = request.match_info.get("url")

    if not target_url or target_url == "":
        return web.Response(text="CORS Proxy Active", status=200)

    parsed = urlparse(target_url)
    if not all([parsed.scheme, parsed.netloc]):
        return web.Response(
            text=f"Invalid URL: '{target_url}'. Ensure it includes the scheme (e.g., https://).",
            status=400,
        )

    host_header = request.host  # e.g., 'localhost:8080' or 'proxy.com'
    if host_header in target_url:
        return web.Response(text="Recursive proxy calls are forbidden.", status=403)

    # 2. Handle CORS Preflight (OPTIONS)
    # We intercept this to tell the browser "Yes, we allow everything."
    if request.method == "OPTIONS":
        return web.Response(status=200, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400",
        })
    # Simple "Request Received" log
    logger.info("--> %s %s", request.method, target_url)

    # 3. Prepare headers to forward
    # We strip 'Host' because the target server expects its own host header
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP and k.lower() != "host"}

    try:
        session: ClientSession = request.app["client_session"]
        # 4. Timeout Strategy
        # total=None: Allow streams to run for hours if needed.
        # connect=10: Kill if we can't connect to target within 10s.
        # sock_read=60: Kill if the server stops sending data for 60s.
        timeout = ClientTimeout(total=None, connect=10, sock_read=60)

        async with session.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=request.content if request.can_read_body else None,
            allow_redirects=True,
            timeout=timeout,
        ) as target_resp:
            # Simple "Response Sent" log
            logger.info("<-- %s %s", target_resp.status, target_url)
            return await process_response(target_resp, request)
    except InvalidURL:
        err, status = "Invalid URL", 400
    except ClientConnectorError:
        err, status = "Target host unreachable.", 504
    except Exception as e:
        logger.exception("Proxy error:")
        err, status = str(e), 502
    return web.Response(text=err, status=status)


async def on_startup(app):
    # Reuse a single session for all outgoing requests
    connector = TCPConnector(limit=0)
    app["client_session"] = ClientSession(connector=connector, auto_decompress=False)
    logger.info("ClientSession created")


async def on_cleanup(app):
    await app["client_session"].close()
    logger.info("ClientSession closed")


def main():
    app = web.Application(client_max_size=1024**3 * 5)  # 5GB, pls don't abuse...
    # Capture everything after the first slash as the URL
    app.router.add_route("*", "/{url:.*}", proxy_handler)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    web.run_app(app, port=PORT, access_log=None)


if __name__ == "__main__":
    main()
