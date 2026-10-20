import socket
import pytest


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """Any attempted network connection fails: tests require no remote service."""
    def blocked(*args, **kwargs):
        raise AssertionError('network access is forbidden in the test suite')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)
