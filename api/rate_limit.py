"""Shared `slowapi` limiter instance.

A module of its own (rather than defined in `main.py`) so `api/routers/predict.py`
can import and apply `@limiter.limit(...)` without a circular import against
the app factory that also needs the same instance to register the
exception handler and middleware (project_refactor.md §16: rate limiting
on `/v1/predict` specifically, in-process via `slowapi`, no Redis).
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
