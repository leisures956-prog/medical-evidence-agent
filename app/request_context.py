"""Request-scoped context shared by API middleware and service logs."""

from __future__ import annotations

from contextvars import ContextVar


request_id_context: ContextVar[str] = ContextVar(
    "request_id_context",
    default="-",
)


def get_request_id() -> str:
    return request_id_context.get()


__all__ = ["get_request_id", "request_id_context"]
