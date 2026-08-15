from __future__ import annotations

from src.auth import AuthManager, UserStore


def test_register_and_authenticate(tmp_path):
    store = UserStore(tmp_path / "users.json")
    store.register("alice", "secret123", role="admin")
    assert store.verify("alice", "secret123") is True
    assert store.verify("alice", "wrong") is False
    assert store.verify("bob", "secret123") is False


def test_password_stored_hashed(tmp_path):
    store = UserStore(tmp_path / "users.json")
    store.register("alice", "secret123")
    raw = (tmp_path / "users.json").read_text(encoding="utf-8")
    assert "secret123" not in raw


def test_authenticate_returns_user(tmp_path):
    store = UserStore(tmp_path / "users.json")
    store.register("alice", "secret123", role="recruiter")
    user = store.authenticate("alice", "secret123")
    assert user is not None
    assert user.username == "alice"
    assert user.role == "recruiter"
    assert store.authenticate("alice", "nope") is None


def test_duplicate_registration_rejected(tmp_path):
    store = UserStore(tmp_path / "users.json")
    store.register("alice", "secret123")
    try:
        store.register("alice", "other")
        assert False, "should reject duplicates"
    except ValueError:
        pass


def test_ensure_default_user(tmp_path):
    store = UserStore(tmp_path / "users.json")
    user = store.ensure_default_user()
    assert user.username == "demo"
    assert store.verify("demo", "demo123") is True
