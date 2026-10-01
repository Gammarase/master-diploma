"""
HTTP Basic authentication middleware.

A pure ASGI middleware, so long-polled requests stream through untouched.
Credentials are compared in constant time.
"""

from __future__ import annotations

import base64
import binascii
import json
import secrets
from collections.abc import Iterable

from starlette.types import ASGIApp, Receive, Scope, Send

__all__ = ["BasicAuthMiddleware"]

_REALM = 'Basic realm="Disinformation Detection", charset="UTF-8"'
_BODY = json.dumps({"detail": "Authentication required."}).encode()


def _credentials(scope: Scope) -> tuple[bytes, bytes] | None:
    """Username and password from the ``Authorization`` header, or None."""
    for name, value in scope.get("headers", ()):
        if name != b"authorization":
            continue
        scheme, _, encoded = value.partition(b" ")
        if scheme.lower() != b"basic":
            return None
        try:
            decoded = base64.b64decode(encoded.strip(), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return None
        username, sep, password = decoded.partition(":")
        if not sep:
            return None
        return username.encode(), password.encode()
    return None


class BasicAuthMiddleware:
    """Require HTTP Basic credentials on every HTTP path except ``exempt_paths``.

    Args:
        app: The wrapped ASGI app.
        username: Expected username.
        password: Expected password.
        exempt_paths: Paths served without credentials (e.g. the health check).
    """

    def __init__(
        self,
        app: ASGIApp,
        username: str,
        password: str,
        exempt_paths: Iterable[str] = ("/api/v1/health",),
    ) -> None:
        self._app = app
        self._username = username.encode()
        self._password = password.encode()
        self._exempt = frozenset(exempt_paths)

    def _authorized(self, scope: Scope) -> bool:
        creds = _credentials(scope)
        if creds is None:
            return False
        # Compare both parts every time: no early exit on the username.
        user_ok = secrets.compare_digest(creds[0], self._username)
        pass_ok = secrets.compare_digest(creds[1], self._password)
        return user_ok and pass_ok

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["path"] in self._exempt
            or self._authorized(scope)
        ):
            await self._app(scope, receive, send)
            return
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_BODY)).encode()),
                    (b"www-authenticate", _REALM.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _BODY})
