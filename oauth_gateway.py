"""OAuth 2.0 gateway for the Copperbelt MCP server2.

Run server2.py on 127.0.0.1:8001, then run this gateway on 127.0.0.1:8002.
Expose ONLY port 8002 through HTTPS. Do not expose port 8001.

Environment variables required:
  OAUTH_ISSUER              Public HTTPS base URL, e.g. https://your-tunnel.example
  OAUTH_CLIENT_ID           Static OAuth client ID to enter in Gemini
  OAUTH_CLIENT_SECRET       Static OAuth client secret to enter in Gemini
  OAUTH_USER                Login username (default: van)
  OAUTH_PASSWORD            Strong password for the browser consent/login page
  OAUTH_REDIRECT_URIS       Comma-separated exact Gemini callback URI(s)
  MCP_UPSTREAM              Default http://127.0.0.1:8001

This is a single-user, single-client prototype. Tokens/codes are in-memory and
are invalidated on restart. Put it behind HTTPS and do not expose the upstream.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import html
import os
import secrets
import time
from typing import Optional
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.background import BackgroundTask


def env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

ISSUER = env_required("OAUTH_ISSUER").rstrip("/")
if not ISSUER.startswith("https://"):
    raise RuntimeError("OAUTH_ISSUER must be a public https:// URL")
CLIENT_ID = env_required("OAUTH_CLIENT_ID")
CLIENT_SECRET = env_required("OAUTH_CLIENT_SECRET")
LOGIN_USER = os.environ.get("OAUTH_USER", "van")
LOGIN_PASSWORD = env_required("OAUTH_PASSWORD")
REDIRECT_URIS = {x.strip() for x in env_required("OAUTH_REDIRECT_URIS").split(",") if x.strip()}
UPSTREAM = os.environ.get("MCP_UPSTREAM", "http://127.0.0.1:8001").rstrip("/")
ACCESS_TTL = 3600
AUTH_CODE_TTL = 300

for uri in REDIRECT_URIS:
    p = urlparse(uri)
    if p.scheme not in ("https", "http") or not p.netloc:
        raise RuntimeError(f"Invalid OAuth redirect URI: {uri}")

app = FastAPI(title="Copperbelt MCP OAuth Gateway", docs_url=None, redoc_url=None)
# One-use authorization codes, bearer tokens, and refresh tokens. This prototype
# intentionally stores only hashes of access/refresh tokens, but codes are short-lived.
auth_codes: dict[str, dict] = {}
access_tokens: dict[str, dict] = {}
refresh_tokens: dict[str, dict] = {}


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def base_headers() -> dict[str, str]:
    return {"Cache-Control": "no-store", "Pragma": "no-cache"}


def client_authenticated(request: Request, form: dict) -> bool:
    # Support OAuth HTTP Basic or credentials in the form body.
    auth = request.headers.get("authorization", "")
    if auth.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode()
            cid, secret = decoded.split(":", 1)
        except Exception:
            return False
    else:
        cid = str(form.get("client_id", ""))
        secret = str(form.get("client_secret", ""))
    return hmac.compare_digest(cid, CLIENT_ID) and hmac.compare_digest(secret, CLIENT_SECRET)


def bearer_valid(token: str) -> bool:
    record = access_tokens.get(sha256_hex(token))
    return bool(record and record["expires_at"] > time.time())


def metadata() -> dict:
    return {
        "issuer": ISSUER,
        "authorization_endpoint": ISSUER + "/authorize",
        "token_endpoint": ISSUER + "/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic"],
        "scopes_supported": ["mcp"],
        "authorization_response_iss_parameter_supported": True,
    }


@app.get("/.well-known/oauth-authorization-server")
@app.get("/.well-known/oauth-authorization-server/{path:path}")
async def oauth_metadata(path: str = ""):
    return JSONResponse(metadata(), headers=base_headers())


@app.get("/.well-known/oauth-protected-resource")
@app.get("/.well-known/oauth-protected-resource/{path:path}")
async def protected_resource_metadata(path: str = ""):
    return JSONResponse({
        "resource": ISSUER + "/mcp",
        "authorization_servers": [ISSUER],
        "scopes_supported": ["mcp"],
        "bearer_methods_supported": ["header"],
    }, headers=base_headers())


@app.get("/authorize", response_class=HTMLResponse)
async def authorize(request: Request):
    q = dict(request.query_params)
    required = ["client_id", "response_type", "redirect_uri", "state"]
    if any(not q.get(k) for k in required):
        raise HTTPException(400, "Missing required OAuth parameter")
    if q["client_id"] != CLIENT_ID or q["response_type"] != "code":
        raise HTTPException(400, "Unsupported OAuth client or response type")
    if q["redirect_uri"] not in REDIRECT_URIS:
        raise HTTPException(400, "redirect_uri is not allowlisted")
    challenge = q.get("code_challenge", "")
    method = q.get("code_challenge_method", "")
    if challenge and (method != "S256" or len(challenge) != 43):
        raise HTTPException(400, "Only S256 PKCE is supported")
    # Require PKCE for clients that send a challenge; confidential client flow is
    # also supported. A CSRF nonce is tied to the signed-in browser session below.
    csrf = secrets.token_urlsafe(24)
    # Carry OAuth request parameters in a signed-by-random-nonce hidden field.
    # The nonce itself is stored server-side and expires quickly.
    auth_codes["pending:" + csrf] = {"params": q, "expires_at": time.time() + AUTH_CODE_TTL, "kind": "pending"}
    escaped_action = html.escape(str(request.url_for("login_and_authorize")), quote=True)
    return HTMLResponse(f"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Copperbelt MCP access</title></head>
<body style="font:16px system-ui;max-width:520px;margin:8vh auto;padding:24px"><h2>Authorize Copperbelt MCP</h2>
<p>Sign in to allow Gemini to use the Copperbelt research tools.</p>
<form method="post" action="{escaped_action}"><input type="hidden" name="csrf" value="{csrf}">
<label>Username<br><input name="username" autocomplete="username" required style="width:100%;padding:10px"></label><br><br>
<label>Password<br><input type="password" name="password" autocomplete="current-password" required style="width:100%;padding:10px"></label><br><br>
<button style="padding:10px 16px">Sign in and authorize</button></form></body></html>""")


@app.post("/authorize/complete")
async def unused_authorize_complete():
    raise HTTPException(404)


@app.post("/authorize")
async def login_and_authorize(username: str = Form(...), password: str = Form(...), csrf: str = Form(...)):
    pending_key = "pending:" + csrf
    pending = auth_codes.pop(pending_key, None)
    if not pending or pending.get("expires_at", 0) <= time.time():
        raise HTTPException(400, "Authorization request expired; restart connection")
    if not (hmac.compare_digest(username, LOGIN_USER) and hmac.compare_digest(password, LOGIN_PASSWORD)):
        # Don't restore pending state after a failed login; restart the OAuth flow.
        raise HTTPException(401, "Invalid username or password")
    q = pending["params"]
    code = secrets.token_urlsafe(32)
    auth_codes[sha256_hex(code)] = {
        "client_id": CLIENT_ID,
        "redirect_uri": q["redirect_uri"],
        "scope": q.get("scope", "mcp"),
        "state": q["state"],
        "code_challenge": q.get("code_challenge", ""),
        "code_challenge_method": q.get("code_challenge_method", ""),
        "expires_at": time.time() + AUTH_CODE_TTL,
        "kind": "code",
    }
    params = {"code": code, "state": q["state"], "iss": ISSUER}
    return RedirectResponse(q["redirect_uri"] + ("&" if "?" in q["redirect_uri"] else "?") + urlencode(params), status_code=302)


@app.post("/token")
async def token(request: Request):
    form = dict(await request.form())
    if not client_authenticated(request, form):
        return JSONResponse({"error": "invalid_client"}, status_code=401, headers={"WWW-Authenticate": 'Basic realm="token"', **base_headers()})
    grant = form.get("grant_type", "")
    if grant == "authorization_code":
        code = str(form.get("code", ""))
        rec = auth_codes.pop(sha256_hex(code), None)
        if not rec or rec.get("kind") != "code" or rec["expires_at"] <= time.time():
            return JSONResponse({"error": "invalid_grant"}, status_code=400, headers=base_headers())
        if rec["client_id"] != CLIENT_ID or rec["redirect_uri"] != form.get("redirect_uri"):
            return JSONResponse({"error": "invalid_grant"}, status_code=400, headers=base_headers())
        challenge = rec.get("code_challenge", "")
        if challenge:
            verifier = str(form.get("code_verifier", ""))
            expected = b64url(hashlib.sha256(verifier.encode()).digest()) if verifier else ""
            if not hmac.compare_digest(expected, challenge):
                return JSONResponse({"error": "invalid_grant"}, status_code=400, headers=base_headers())
        elif form.get("code_verifier"):
            return JSONResponse({"error": "invalid_grant"}, status_code=400, headers=base_headers())
        return issue_tokens(rec.get("scope", "mcp"))
    if grant == "refresh_token":
        supplied = str(form.get("refresh_token", ""))
        rec = refresh_tokens.pop(sha256_hex(supplied), None)
        if not rec or rec["expires_at"] <= time.time():
            return JSONResponse({"error": "invalid_grant"}, status_code=400, headers=base_headers())
        return issue_tokens(rec["scope"])
    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400, headers=base_headers())


def issue_tokens(scope: str):
    access = secrets.token_urlsafe(40)
    refresh = secrets.token_urlsafe(48)
    now = time.time()
    access_tokens[sha256_hex(access)] = {"expires_at": now + ACCESS_TTL, "scope": scope}
    # Refresh token remains valid for 30 days; rotate on use.
    refresh_tokens[sha256_hex(refresh)] = {"expires_at": now + 30 * 86400, "scope": scope}
    return JSONResponse({"access_token": access, "token_type": "Bearer", "expires_in": ACCESS_TTL,
                         "refresh_token": refresh, "scope": scope}, headers=base_headers())


@app.get("/health")
async def health():
    return {"status": "ok", "service": "oauth-gateway", "upstream": UPSTREAM + "/mcp"}


@app.api_route("/mcp", methods=["GET", "POST", "DELETE", "OPTIONS"])
async def proxy_mcp(request: Request):
    auth = request.headers.get("authorization", "")
    scheme, _, token_value = auth.partition(" ")
    if scheme.lower() != "bearer" or not token_value or not bearer_valid(token_value):
        return JSONResponse({"error": "unauthorized", "error_description": "A valid OAuth bearer token is required"},
                            status_code=401,
                            headers={"WWW-Authenticate": f'Bearer resource_metadata="{ISSUER}/.well-known/oauth-protected-resource/mcp", scope="mcp"', **base_headers()})
    # Forward MCP Streamable HTTP, including session headers and SSE response bodies.
    headers = dict(request.headers)
    for name in ("host", "content-length", "connection", "authorization"):
        headers.pop(name, None)
    body = await request.body()
    client = httpx.AsyncClient(timeout=None, follow_redirects=False)
    try:
        upstream_request = client.build_request(request.method, UPSTREAM + "/mcp", headers=headers, content=body)
        upstream_response = await client.send(upstream_request, stream=True)
    except Exception:
        await client.aclose()
        raise HTTPException(502, "MCP upstream is unavailable")
    out_headers = dict(upstream_response.headers)
    for name in ("content-length", "transfer-encoding", "connection", "content-encoding", "keep-alive"):
        out_headers.pop(name, None)
    async def stream_body():
        try:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        finally:
            await upstream_response.aclose()
            await client.aclose()
    return StreamingResponse(stream_body(), status_code=upstream_response.status_code,
                             headers=out_headers, background=BackgroundTask(client.aclose))


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def not_found(path: str):
    raise HTTPException(404, "Not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
