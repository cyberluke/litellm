from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, Protocol
from urllib.parse import urlparse

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.caching.redis_cache import RedisCache
from litellm.proxy._experimental.mcp_server.oauth_utils import get_request_base_url, validate_redirect_uri_shape
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

HOSTED_ACCESS_PREFIX: Final = "llm_hosted_"
HOSTED_REFRESH_PREFIX: Final = "llm_hrefresh_"
HOSTED_ACCESS_TTL: Final = 300
HOSTED_GRANT_TTL: Final = 86400
MAX_HOSTED_REFRESHES: Final = 512
HOSTED_SCOPE: Final = "proxy:read"
_MODEL_ROUTES: Final = frozenset({"/models", "/v1/models"})
_REPORT_ROUTES: Final = frozenset(
    {"/global/activity", "/global/activity/model", "/global/spend", "/global/spend/provider", "/global/spend/report"}
)
_TOKEN_SHAPE: Final = re.compile(r"[A-Za-z0-9_-]{32}\.[A-Za-z0-9_-]{43}\Z")


def hosted_redirect_is_allowed(redirect_uri: str) -> bool:
    try:
        parsed: Final = urlparse(redirect_uri)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            return False
        if "?" in redirect_uri or "#" in redirect_uri:
            return False
        validate_redirect_uri_shape(parsed)
    except (HTTPException, ValueError):
        return False
    return redirect_uri in tuple(
        entry.strip() for entry in os.environ.get("LITELLM_PROXY_API_OAUTH_REDIRECT_URIS", "").split(",")
    )


class HostedFailure(BaseModel):
    model_config = ConfigDict(frozen=True)
    error: Literal["invalid_grant", "temporarily_unavailable", "insufficient_scope"] = "invalid_grant"
    description: str = "the hosted application grant is no longer valid; sign in again"


class HostedGrant(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    grant_id: str
    user_id: str
    team_id: str | None
    client_id: str
    redirect_uri: str
    resource: str
    expires_at: int
    access_expires_at: int
    access_hash: str
    refresh_hash: str
    refresh_count: int = Field(default=0, ge=0, le=MAX_HOSTED_REFRESHES)


class HostedTokens(BaseModel):
    model_config = ConfigDict(frozen=True)
    access_token: str = Field(repr=False)
    refresh_token: str = Field(repr=False)
    user_id: str
    team_id: str | None
    expires_in: int
    token_type: Literal["Bearer"] = "Bearer"
    scope: Literal["proxy:read"] = HOSTED_SCOPE


class HostedGrantStore(Protocol):
    async def read(self, grant_id: str) -> HostedGrant | HostedFailure: ...

    async def replace(self, grant: HostedGrant, previous: HostedGrant | None, ttl: int) -> bool: ...

    async def delete(self, grant_id: str) -> None: ...

    async def revoke_replayed(self, grant_id: str, token_hash: str) -> None: ...


_REPLACE_GRANT: Final = """
local current = redis.call('GET', KEYS[1])
if ARGV[1] == '' and redis.call('EXISTS', KEYS[2]) == 1 then
    return 0
end
if (ARGV[1] == '' and not current) or current == ARGV[1] then
    redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
    redis.call('SET', KEYS[2], '1', 'EX', ARGV[3])
    if ARGV[4] ~= '' then
        redis.call('SADD', KEYS[3], ARGV[4])
        redis.call('EXPIRE', KEYS[3], ARGV[3])
    end
    return 1
end
return 0
"""
_REVOKE_REPLAYED: Final = """
if redis.call('SISMEMBER', KEYS[2], ARGV[1]) == 1 then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class RedisHostedGrantStore:
    def __init__(self, cache: RedisCache, master_key: str) -> None:
        self._cache: Final = cache
        self._gateway: Final = hashlib.sha256(master_key.encode()).hexdigest()

    def _key(self, grant_id: str) -> str:
        return self._cache.check_and_fix_namespace(f"hosted_proxy_grant:{{{self._gateway}:{grant_id}}}")

    async def read(self, grant_id: str) -> HostedGrant | HostedFailure:
        value: Final = await self._cache.async_eval("return redis.call('GET', KEYS[1])", 1, self._key(grant_id))
        if not isinstance(value, (str, bytes)):
            return HostedFailure()
        try:
            return HostedGrant.model_validate_json(value)
        except ValidationError:
            return HostedFailure()

    async def replace(self, grant: HostedGrant, previous: HostedGrant | None, ttl: int) -> bool:
        result: Final = await self._cache.async_eval(
            _REPLACE_GRANT,
            3,
            self._key(grant.grant_id),
            self._key(grant.grant_id) + ":issued",
            self._key(grant.grant_id) + ":spent",
            previous.model_dump_json() if previous is not None else "",
            grant.model_dump_json(),
            ttl,
            previous.refresh_hash if previous is not None else "",
        )
        return isinstance(result, int) and result == 1

    async def delete(self, grant_id: str) -> None:
        await self._cache.async_eval("return redis.call('DEL', KEYS[1])", 1, self._key(grant_id))

    async def revoke_replayed(self, grant_id: str, token_hash: str) -> None:
        await self._cache.async_eval(
            _REVOKE_REPLAYED, 2, self._key(grant_id), self._key(grant_id) + ":spent", token_hash
        )


class LoadHostedUser(Protocol):
    async def __call__(self, user_id: str, team_id: str | None, /) -> UserAPIKeyAuth | HostedFailure: ...


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _grant_id(token: str, prefix: str) -> str | None:
    if not token.startswith(prefix):
        return None
    suffix: Final = token.removeprefix(prefix)
    return suffix.split(".", 1)[0] if _TOKEN_SHAPE.fullmatch(suffix) else None


def _unavailable() -> HostedFailure:
    return HostedFailure(error="temporarily_unavailable", description="hosted application authorization is unavailable")


class HostedProxyAuth:
    def __init__(self, store: HostedGrantStore, load_user: LoadHostedUser, now: datetime) -> None:
        self._store: Final = store
        self._load_user: Final = load_user
        self._now: Final = int(now.timestamp())

    async def issue(
        self,
        user_id: str,
        team_id: str | None,
        client_id: str,
        redirect_uri: str,
        resource: str,
        code_id: str,
    ) -> HostedTokens | HostedFailure:
        if not hosted_redirect_is_allowed(redirect_uri):
            return HostedFailure()
        identity: Final = await self._load_user(user_id, team_id)
        if isinstance(identity, HostedFailure):
            return identity
        return await self._mint(
            HostedGrant(
                grant_id=hashlib.sha256(code_id.encode()).hexdigest()[:32],
                user_id=user_id,
                team_id=team_id,
                client_id=client_id,
                redirect_uri=redirect_uri,
                resource=resource,
                expires_at=self._now + HOSTED_GRANT_TTL,
                access_expires_at=0,
                access_hash="",
                refresh_hash="",
            ),
            previous=None,
        )

    async def _read(
        self, token: str, prefix: str, client_id: str | None, resource: str | None
    ) -> HostedGrant | HostedFailure:
        grant_id: Final = _grant_id(token, prefix)
        if grant_id is None:
            return HostedFailure()
        try:
            grant: Final = await self._store.read(grant_id)
        except Exception:  # noqa: BLE001  # authorization must fail closed on every store failure
            return _unavailable()
        if isinstance(grant, HostedFailure):
            return grant
        expected: Final = grant.refresh_hash if prefix == HOSTED_REFRESH_PREFIX else grant.access_hash
        expires: Final = grant.expires_at if prefix == HOSTED_REFRESH_PREFIX else grant.access_expires_at
        if (
            grant.grant_id != grant_id
            or self._now >= expires
            or self._now >= grant.expires_at
            or (client_id is not None and client_id != grant.client_id)
            or (resource is not None and resource != grant.resource)
        ):
            return HostedFailure()
        if not hmac.compare_digest(_digest(token), expected):
            if prefix == HOSTED_REFRESH_PREFIX:
                try:
                    await self._store.revoke_replayed(grant_id, _digest(token))
                except Exception:  # noqa: BLE001  # a failed replay revocation must not be hidden
                    return _unavailable()
            return HostedFailure()
        return grant

    async def refresh(self, token: str, client_id: str, resource: str) -> HostedTokens | HostedFailure:
        grant: Final = await self._read(token, HOSTED_REFRESH_PREFIX, client_id, resource)
        if isinstance(grant, HostedFailure):
            return grant
        if grant.refresh_count >= MAX_HOSTED_REFRESHES:
            return HostedFailure(description="the application's refresh limit was reached; sign in again")
        if not hosted_redirect_is_allowed(grant.redirect_uri):
            return HostedFailure()
        identity: Final = await self._load_user(grant.user_id, grant.team_id)
        if isinstance(identity, HostedFailure):
            return identity
        return await self._mint(grant, previous=grant)

    async def _mint(self, grant: HostedGrant, previous: HostedGrant | None) -> HostedTokens | HostedFailure:
        access: Final = f"{HOSTED_ACCESS_PREFIX}{grant.grant_id}.{secrets.token_urlsafe(32)}"
        refresh: Final = f"{HOSTED_REFRESH_PREFIX}{grant.grant_id}.{secrets.token_urlsafe(32)}"
        expires_in: Final = min(HOSTED_ACCESS_TTL, grant.expires_at - self._now)
        updated: Final = grant.model_copy(
            update=MappingProxyType(
                {
                    "access_hash": _digest(access),
                    "refresh_hash": _digest(refresh),
                    "access_expires_at": self._now + expires_in,
                    "refresh_count": grant.refresh_count + 1 if previous is not None else 0,
                }
            )
        )
        try:
            stored: Final = await self._store.replace(updated, previous, grant.expires_at - self._now)
        except Exception:  # noqa: BLE001  # never issue a credential whose shared grant was not stored
            return _unavailable()
        if not stored:
            if previous is not None:
                try:
                    await self._store.revoke_replayed(grant.grant_id, previous.refresh_hash)
                except Exception:  # noqa: BLE001  # concurrent refresh replay invalidates the grant or fails closed
                    return _unavailable()
            return HostedFailure()
        return HostedTokens(
            access_token=access,
            refresh_token=refresh,
            expires_in=expires_in,
            user_id=grant.user_id,
            team_id=grant.team_id,
        )

    async def authenticate(self, token: str, resource: str, route: str, method: str) -> UserAPIKeyAuth | HostedFailure:
        if method != "GET" or route not in _MODEL_ROUTES | _REPORT_ROUTES:
            return HostedFailure(
                error="insufficient_scope", description="this application has read-only model and usage access"
            )
        grant: Final = await self._read(token, HOSTED_ACCESS_PREFIX, None, resource)
        if isinstance(grant, HostedFailure):
            return grant
        if not hosted_redirect_is_allowed(grant.redirect_uri):
            return HostedFailure()
        identity: Final = await self._load_user(grant.user_id, grant.team_id)
        if isinstance(identity, HostedFailure):
            return identity
        if route in _REPORT_ROUTES and identity.user_role not in (
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        ):
            return HostedFailure(
                error="insufficient_scope", description="global usage requires a current proxy administrator role"
            )
        return identity.model_copy(update=MappingProxyType({"token": _digest(token), "api_key": _digest(token)}))

    async def revoke(self, token: str, client_id: str) -> HostedFailure | None:
        prefix: Final = HOSTED_REFRESH_PREFIX if token.startswith(HOSTED_REFRESH_PREFIX) else HOSTED_ACCESS_PREFIX
        grant: Final = await self._read(token, prefix, client_id, None)
        if isinstance(grant, HostedFailure):
            return grant if grant.error == "temporarily_unavailable" else None
        try:
            await self._store.delete(grant.grant_id)
        except Exception:  # noqa: BLE001  # a failed revocation must be reported as retryable
            return _unavailable()
        return None


async def load_hosted_user(user_id: str, team_id: str | None) -> UserAPIKeyAuth | HostedFailure:
    from litellm.proxy._experimental.mcp_server.bridge_token_flow import (
        load_active_user_by_id,  # noqa: PLC0415  # proxy import cycle
    )
    from litellm.proxy.auth.auth_checks import (  # noqa: PLC0415  # proxy import cycle
        effective_user_role,
        get_team_object,
    )
    from litellm.proxy.auth.resolvers.grants import user_models  # noqa: PLC0415  # shared typed user grant projection
    from litellm.proxy.auth.team_grants import team_grants  # noqa: PLC0415  # shared team permission projection
    from litellm.proxy.management_endpoints.ui_sso import (
        fetch_cli_sso_team_details,  # noqa: PLC0415  # existing live team selection rules
    )
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # startup-owned dependencies
        prisma_client,
        user_api_key_cache,
    )

    user: Final = await load_active_user_by_id(user_id, source="database")
    if isinstance(user, str):
        return _unavailable() if user in ("unavailable", "unresolvable", "faulted") else HostedFailure()
    if team_id is None and user.teams:
        if prisma_client is None:
            return _unavailable()
        teams: Final = await fetch_cli_sso_team_details(prisma_client, user.teams)
        if teams is None:
            return _unavailable()
        if any(team.team_id is not None for team in teams):
            return HostedFailure(description="select a current team and sign in again")
    if team_id is not None and team_id not in user.teams:
        return HostedFailure()
    try:
        team: Final = (
            await get_team_object(
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                check_db_only=True,
            )
            if team_id is not None
            else None
        )
    except Exception:  # noqa: BLE001  # no stale membership or permissions on lookup failure
        return _unavailable()
    if team is not None and team.blocked:
        return HostedFailure()
    return UserAPIKeyAuth.model_validate(
        MappingProxyType(
            {
                "user_id": user.user_id,
                "user_role": effective_user_role(user.user_role),
                "team_id": team_id,
                "models": user_models(user) if team is None else (),
                **team_grants(team, None, user.user_id),
            }
        )
    )


def hosted_proxy_auth() -> HostedProxyAuth | HostedFailure:
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # startup-owned shared authority
        master_key,
        redis_usage_cache,
    )

    if redis_usage_cache is None or not isinstance(master_key, str) or not master_key:
        return _unavailable()
    return HostedProxyAuth(
        RedisHostedGrantStore(redis_usage_cache, master_key), load_hosted_user, datetime.now(timezone.utc)
    )


async def authenticate_hosted_request(request: Request, token: str, route: str) -> UserAPIKeyAuth:
    service: Final = hosted_proxy_auth()
    result: Final = (
        service
        if isinstance(service, HostedFailure)
        else await service.authenticate(
            token,
            get_request_base_url(request),
            route,
            request.method,
        )
    )
    if isinstance(result, HostedFailure):
        status: Final = (
            503 if result.error == "temporarily_unavailable" else 403 if result.error == "insufficient_scope" else 401
        )
        raise HTTPException(status_code=status, detail=result.description)
    return result
