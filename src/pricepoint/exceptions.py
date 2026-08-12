"""Domain exception hierarchy.

Small and deliberately flat (project_refactor.md §8.2): a handful of
exception types that `api/main.py`'s FastAPI exception handlers map to
HTTP status codes in one place, rather than every route handler doing its
own ad hoc ``try/except`` -> ``HTTPException`` translation. Raised from
``pricepoint``/``api/services`` code that has no HTTP concerns of its
own -- these exceptions are plain Python exceptions, importable and
raisable from CLI code (``run.py``) just as validly as from the API.
"""

from __future__ import annotations

from typing import Any


class PricePointError(Exception):
    """Base class for all domain errors raised by this project.

    Parameters
    ----------
    message : str
        Human-readable error description.
    context : dict, optional
        Structured extra detail (e.g. the valid date range for a
        `DataValidationError`) -- surfaced verbatim in the API's error
        response body so a client can act on it programmatically, not
        just display the message.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


class DataValidationError(PricePointError):
    """A request or input fails a domain-level validation rule.

    Distinct from Pydantic's own type/shape validation (which FastAPI
    handles automatically) -- this is for rules that depend on the
    actual data (e.g. "date must fall within the dataset's observed
    range", project_refactor.md §8.2), which Pydantic's static schema
    can't express on its own. Maps to HTTP 422.
    """


class ArtifactNotFoundError(PricePointError):
    """A referenced product, artifact, or resource does not exist.

    Covers both "this canonical product isn't in the mart" and "this
    precomputed artifact file is missing from disk" -- both are the same
    shape of problem from the caller's perspective (the thing you asked
    for isn't there). Maps to HTTP 404.
    """
