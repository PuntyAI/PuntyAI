"""Tests for monitor start/stop auth guards."""

import pytest
from unittest.mock import MagicMock, patch

from punty.api.results import start_monitor, stop_monitor


class FakeRequest:
    """Minimal request stub with session support."""

    def __init__(self, user=None):
        self._session = {"user": user} if user else {}
        self.app = MagicMock()
        self.app.state.results_monitor = MagicMock()

    @property
    def session(self):
        return self._session


@pytest.mark.asyncio
class TestMonitorAuth:
    async def test_start_monitor_no_session_returns_401(self):
        req = FakeRequest(user=None)
        resp = await start_monitor(req)
        assert resp.status_code == 401

    async def test_stop_monitor_no_session_returns_401(self):
        req = FakeRequest(user=None)
        resp = await stop_monitor(req)
        assert resp.status_code == 401

    async def test_start_monitor_with_session_succeeds(self):
        req = FakeRequest(user="test@example.com")
        resp = await start_monitor(req)
        assert resp == {"status": "started"}

    async def test_stop_monitor_with_session_succeeds(self):
        req = FakeRequest(user="test@example.com")
        resp = await stop_monitor(req)
        assert resp == {"status": "stopped"}
