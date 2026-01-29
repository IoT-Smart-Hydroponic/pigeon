import json
import time
import tempfile
from pathlib import Path
from collections import deque
from threading import Lock
from typing import Any

import requests

from .config import config
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(
        self,
        max_messages: int = 30,
        per_seconds: float = 60.0,
        burst_messages: int = 5,
        burst_seconds: float = 2.0,
    ) -> None:
        self.max_messages = max_messages
        self.per_seconds = per_seconds
        self.burst_messages = burst_messages
        self.burst_seconds = burst_seconds
        self._window = deque()
        self._burst = deque()
        self._lock = Lock()

    def _prune(self, now: float) -> None:
        while self._window and now - self._window[0] >= self.per_seconds:
            self._window.popleft()
        while self._burst and now - self._burst[0] >= self.burst_seconds:
            self._burst.popleft()

    def wait_for_slot(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)

                window_ok = len(self._window) < self.max_messages
                burst_ok = len(self._burst) < self.burst_messages

                if window_ok and burst_ok:
                    self._window.append(now)
                    self._burst.append(now)
                    return

                wait_window = (
                    self.per_seconds - (now - self._window[0])
                    if not window_ok and self._window
                    else None
                )
                wait_burst = (
                    self.burst_seconds - (now - self._burst[0])
                    if not burst_ok and self._burst
                    else None
                )

                wait_for = min(w for w in [wait_window, wait_burst] if w is not None)

            if wait_for > 0:
                time.sleep(wait_for)


class DiscordNotifier:
    def __init__(self):
        self.webhook_url = config.WEBHOOK_DEVLOGS_CHANNEL
        self.base_dir = Path(tempfile.gettempdir()) / "pigeon"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._state_cache: dict[str, dict[str, str | None]] = {}
        self.rate_limiter = RateLimiter()
        self.session = requests.Session()
        self.timeout = 10

    def _scope_key(self, scope_key: str) -> str:
        return scope_key.replace("/", "__").replace("\\", "__")

    def _state_path(self, scope_key: str) -> Path:
        return self.base_dir / f"pigeon_temp_msg_id_{self._scope_key(scope_key)}.json"

    def _load_state(self, scope_key: str) -> dict[str, str | None]:
        if scope_key in self._state_cache:
            return self._state_cache[scope_key]

        state = {"message_id": None, "event": None}
        file_path = self._state_path(scope_key)
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8").strip()
                if content:
                    data = json.loads(content)
                    state["message_id"] = data.get("message_id")
                    state["event"] = data.get("event")
            except Exception:
                state = {"message_id": None, "event": None}

        self._state_cache[scope_key] = state
        return state

    def _save_state(self, scope_key: str, message_id: str, event: str) -> None:
        state = {"message_id": message_id, "event": event}
        self._state_cache[scope_key] = state
        self._state_path(scope_key).write_text(json.dumps(state), encoding="utf-8")

    def _clear_state(self, scope_key: str) -> None:
        self._state_cache.pop(scope_key, None)
        file_path = self._state_path(scope_key)
        if file_path.exists():
            file_path.unlink()

    def _request_with_retry(
        self, method: str, url: str, payload: Any
    ) -> requests.Response:
        max_retries = 3
        last_error = None

        for _ in range(max_retries):
            self.rate_limiter.wait_for_slot()
            response = self.session.request(
                method, url, json=payload, timeout=self.timeout
            )

            if response.status_code != 429:
                return response

            retry_after = None
            try:
                body = response.json()
                retry_after = body.get("retry_after")
            except ValueError:
                retry_after = None

            if retry_after is None:
                retry_after_header = response.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        retry_after = None

            if retry_after is None:
                retry_after = 1.5

            logger.warning(
                "Discord rate limit hit. Retrying after %.2f seconds.", retry_after
            )
            time.sleep(retry_after)
            last_error = response

        if last_error is not None:
            return last_error
        return response

    def has_active_message(self, event: str, scope_key: str = "global") -> bool:
        state = self._load_state(scope_key)
        return state.get("message_id") is not None and state.get("event") == event

    def send_message(
        self,
        payload: Any,
        event: str,
        force: bool = False,
        track: bool = True,
        scope_key: str = "global",
    ) -> None:
        if not self.webhook_url:
            return

        if not force and self.has_active_message(event, scope_key=scope_key):
            return

        response = self._request_with_retry(
            "POST",
            self.webhook_url + "?wait=true",
            payload,
        )
        rate_limit_count = response.headers.get("X-RateLimit-Remaining", "unknown")
        total_rate_limit = response.headers.get("X-RateLimit-Limit", "unknown")

        logger.info(f"Rate Limit: {rate_limit_count}/{total_rate_limit}")

        response.raise_for_status()

        message_data = response.json()
        new_message_id = message_data.get("id")

        if track:
            self._save_state(scope_key, new_message_id, event)
        logger.info("Message sent with ID: %s", new_message_id)

    def edit_last_message(
        self, new_payload: Any, event: str, scope_key: str = "global"
    ) -> None:
        state = self._load_state(scope_key)
        message_id = state.get("message_id")
        logger.info("Message ID to edit: %s", message_id)
        if not self.webhook_url or not message_id:
            return

        if state.get("event") != event:
            self.send_message(
                new_payload, event, scope_key=scope_key
            )  # Fallback to sending a new message if event changed
            return

        edit_url = f"{self.webhook_url}/messages/{message_id}"
        response = self._request_with_retry("PATCH", edit_url, new_payload)
        rate_limit_count = response.headers.get("X-RateLimit-Remaining", "unknown")
        total_rate_limit = response.headers.get("X-RateLimit-Limit", "unknown")

        logger.info(f"Rate Limit: {rate_limit_count}/{total_rate_limit}")
        response.raise_for_status()

    def remove_id_file(self, scope_key: str = "global") -> None:
        self._clear_state(scope_key)
