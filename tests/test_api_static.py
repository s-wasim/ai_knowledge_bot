# tests/test_api_static.py
from fastapi.testclient import TestClient

from app.server import app


class TestStaticFrontend:
    def test_root_serves_the_dc_html_entrypoint(self):
        client = TestClient(app)
        res = client.get("/")

        assert res.status_code == 200
        assert "<x-dc>" in res.text

    def test_support_js_is_served(self):
        client = TestClient(app)
        res = client.get("/support.js")

        assert res.status_code == 200
        assert "dc-runtime" in res.text

    def test_adapter_js_is_served(self):
        client = TestClient(app)
        res = client.get("/src/api/adapter.js")

        assert res.status_code == 200

    def test_api_routes_still_resolve_alongside_static_mount(self):
        client = TestClient(app)
        res = client.get("/config/allowlist")

        assert res.status_code == 200
        assert isinstance(res.json(), list)
