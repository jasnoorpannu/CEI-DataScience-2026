from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.logging_config import get_logger

logger = get_logger("resumefit.auth")

_ITERATIONS = 200_000


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)


@dataclass
class User:
    username: str
    role: str = "recruiter"

    def to_dict(self) -> dict:
        return {"username": self.username, "role": self.role}


class UserStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else config.USERS_FILE

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.chmod(self.path, 0o600)

    def register(self, username: str, password: str, role: str = "recruiter") -> User:
        username = username.strip().lower()
        data = self._load()
        if username in data:
            raise ValueError(f"User '{username}' already exists.")
        salt = secrets.token_bytes(16)
        data[username] = {
            "salt": salt.hex(),
            "hash": _hash_password(password, salt).hex(),
            "role": role,
        }
        self._save(data)
        logger.info("Registered user '%s' (role=%s).", username, role)
        return User(username=username, role=role)

    def verify(self, username: str, password: str) -> bool:
        data = self._load()
        record = data.get(username.strip().lower())
        if not record:
            return False
        salt = bytes.fromhex(record["salt"])
        expected = bytes.fromhex(record["hash"])
        return hmac.compare_digest(_hash_password(password, salt), expected)

    def role(self, username: str) -> str:
        record = self._load().get(username.strip().lower())
        return record.get("role", "recruiter") if record else "recruiter"

    def authenticate(self, username: str, password: str) -> User | None:
        if self.verify(username, password):
            username = username.strip().lower()
            return User(username=username, role=self.role(username))
        return None

    def ensure_default_user(self) -> User:
        data = self._load()
        if "demo" not in data:
            return self.register("demo", "demo123", role="admin")
        return User(username="demo", role=self.role("demo"))


class AuthManager:
    def __init__(self, store: UserStore | None = None) -> None:
        self.store = store or UserStore()
        self._session_key = "resumefit_auth"

    def _current(self) -> User | None:
        import streamlit as st

        user = st.session_state.get(self._session_key)
        return user if isinstance(user, User) else None

    def is_authenticated(self) -> bool:
        return self._current() is not None

    def current_user(self) -> User | None:
        return self._current()

    def login(self, username: str, password: str) -> bool:
        import streamlit as st

        user = self.store.authenticate(username, password)
        if user:
            st.session_state[self._session_key] = user
            logger.info("Login for user '%s'.", user.username)
            return True
        return False

    def logout(self) -> None:
        import streamlit as st

        st.session_state.pop(self._session_key, None)

    def render_login(self) -> bool:
        import streamlit as st

        st.title("Sign in")
        st.caption("Default demo account: `demo` / `demo123` (admin).")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Sign in", type="primary"):
            if self.login(username, password):
                st.success("Signed in.")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        return self.is_authenticated()
