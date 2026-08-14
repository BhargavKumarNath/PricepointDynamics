"""Contract tests for CORS (project_refactor.md §16): the API must allow
only the explicitly configured frontend origin(s), never `*` -- basic
hygiene against casual cross-origin scraping, not enterprise-grade
protection, per this project's own stated threat model.
"""

from __future__ import annotations

from api.dependencies import get_settings


class TestCors:
    def test_configured_origins_are_not_wildcard(self):
        """The actual, live-loaded settings this app runs with -- not just
        config.yaml's text -- must never resolve to `*`."""
        origins = get_settings().api.cors_origins
        assert origins, "cors_origins must not be empty"
        assert "*" not in origins

    def test_allowed_origin_gets_the_cors_header(self, client):
        allowed_origin = get_settings().api.cors_origins[0]
        r = client.get("/health", headers={"Origin": allowed_origin})
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == allowed_origin

    def test_unlisted_origin_does_not_get_the_cors_header(self, client):
        r = client.get("/health", headers={"Origin": "https://not-an-allowed-origin.example.com"})
        assert r.status_code == 200
        assert "access-control-allow-origin" not in r.headers

    def test_preflight_for_unlisted_origin_is_rejected(self, client):
        r = client.options(
            "/v1/predict",
            headers={
                "Origin": "https://not-an-allowed-origin.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in r.headers
