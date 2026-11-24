"""Request pacing for NVIDIA NIM embeddings, shared by every PipesHub process in the container.

An NVIDIA Build key allows a fixed number of requests a minute (40 on the free tier), and the query service
(searches, model checks) and the indexing service (documents, images) are separate processes sending with the
same key. A per-process limiter would let each of them spend the whole allowance, so the pacing state is one
small file per endpoint and key, in the directory the container's processes already share for resource slots
(``shared_slots_directory``), read and rewritten under an exclusive ``flock``.

Two lanes share that state. ``BULK`` (indexing) may make at most ``bulk`` of the sends in any 60 s window and
is spaced evenly; ``INTERACTIVE`` (searches) may use the whole ``total``, so a search is never queued behind a
backlog of documents. Nothing is booked ahead: a caller that cannot send yet sleeps until the earliest moment
it could and asks again, so a cancelled or timed-out caller leaves nothing behind. A caller that would wait
longer than its allowance gets ``PacerBusy`` instead of sleeping through the deadline of the work around it.

Without POSIX file locks (Windows) or a directory only this user can change, each process keeps its own
state and the limit counts per process.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypeVar, cast

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX
    fcntl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

_R = TypeVar("_R")

WINDOW_S = 60.0
# A wait longer than this is a cooldown from an older run (or a clock that jumped back): it is dropped.
_MAX_COOLDOWN_S = 3600.0
# What a caller may sleep in the pacer before it hands the work back. The indexing batch that waits here
# has 120 s for the wait and the request together, and the request may take 60 s.
DEFAULT_MAX_WAIT_S = 45.0
_STATE_DIRECTORY_NAME = "nvidia-nim-pacer"
_MAX_STATE_BYTES = 64 * 1024


class Lane(Enum):
    BULK = "bulk"
    INTERACTIVE = "interactive"


class PacerBusy(Exception):
    """The caller would have to wait ``wait_s`` seconds, more than its allowance."""

    def __init__(self, wait_s: float) -> None:
        super().__init__(f"NVIDIA NIM pacing needs a {wait_s:.0f}s wait")
        self.wait_s = wait_s


@dataclass(frozen=True)
class PacerLimits:
    """Sends allowed in any 60 s window: ``total`` for every caller, ``bulk`` of them for indexing."""

    total: int
    bulk: int

    @classmethod
    def of(cls, total: int, interactive_reserve: int) -> PacerLimits:
        total = max(0, total)
        if total == 0:
            return cls(0, 0)
        return cls(total, max(1, total - max(0, interactive_reserve)))

    @property
    def enabled(self) -> bool:
        return self.total > 0


@dataclass
class _State:
    cooldown_until: float = 0.0
    bulk_next: float = 0.0
    sends: list[float] = field(default_factory=list[float])

    @classmethod
    def load(cls, raw: bytes) -> _State:
        try:
            loaded: object = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except (ValueError, UnicodeDecodeError):
            return cls()
        if not isinstance(loaded, dict):
            return cls()
        data = cast("dict[str, object]", loaded)
        sends = data.get("sends")
        return cls(
            cooldown_until=_number(data.get("cooldown_until")),
            bulk_next=_number(data.get("bulk_next")),
            sends=sorted(_number(s) for s in cast("list[object]", sends)) if isinstance(sends, list) else [],
        )

    def dump(self) -> bytes:
        payload = {"cooldown_until": self.cooldown_until, "bulk_next": self.bulk_next, "sends": self.sends}
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _try_send(state: _State, limits: PacerLimits, lane: Lane, now: float) -> float:
    """Record a send at ``now`` and return 0, or return the seconds until this lane may send."""
    # Entries from another run, or written before the wall clock jumped back, would block every caller.
    if state.cooldown_until > now + _MAX_COOLDOWN_S:
        state.cooldown_until = now
    if state.bulk_next > now + WINDOW_S:
        state.bulk_next = now
    state.sends = [s for s in state.sends if now - WINDOW_S < s <= now + WINDOW_S]

    waits = [state.cooldown_until - now]
    if limits.enabled:
        cap = limits.bulk if lane is Lane.BULK else limits.total
        if len(state.sends) >= cap:
            waits.append(state.sends[-cap] + WINDOW_S - now)
        if lane is Lane.BULK:
            waits.append(state.bulk_next - now)
    wait = max(waits)
    if wait > 0.0:
        return wait
    if limits.enabled:
        state.sends.append(now)
        state.sends.sort()
        if lane is Lane.BULK:
            state.bulk_next = now + WINDOW_S / limits.bulk
    return 0.0


class _MemoryStore:
    """This process's state only."""

    blocking = False

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._state = _State()

    def update(self, fn: Callable[[_State], _R]) -> _R:
        with self._lock:
            return fn(self._state)


class _FileStore:
    """State in a private file every process of this user in the container opens."""

    blocking = True

    def __init__(self, path: Path) -> None:
        from app.services.resource_governor.shared_slots import open_private_file

        super().__init__()
        self._path = path
        self._open = open_private_file
        # flock belongs to the open file, which the threads of one process share: they take turns here first.
        self._lock = threading.Lock()
        self._fd = open_private_file(path)

    def _replaced(self) -> bool:
        """Whether the open file no longer has its name (deleted, as by a /tmp cleaner, or replaced): processes
        started later would open a new file and no longer see this one's sends."""
        try:
            current = os.stat(self._path, follow_symlinks=False)
            held = os.fstat(self._fd)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return held.st_nlink == 0 or (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino)

    def _reopen_if_replaced(self) -> None:
        if not self._replaced():
            return
        from app.services.resource_governor.shared_slots import ensure_private_directory

        try:
            ensure_private_directory(self._path.parent)
            fd = self._open(self._path)
        except OSError:
            # Keep pacing on the file already open; only processes started from now on are not counted with it.
            return
        with contextlib.suppress(OSError):
            os.close(self._fd)
        self._fd = fd

    def update(self, fn: Callable[[_State], _R]) -> _R:
        if fcntl is None:  # pragma: no cover - _open_store builds this only where fcntl exists
            raise RuntimeError("file-backed pacing needs POSIX file locks")
        with self._lock:
            self._reopen_if_replaced()
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            try:
                state = _State.load(os.pread(self._fd, _MAX_STATE_BYTES, 0))
                result = fn(state)
                payload = state.dump()
                _ = os.pwrite(self._fd, payload, 0)
                os.ftruncate(self._fd, len(payload))
                return result
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)


_warned_fallback = False


def _open_store(name: str, directory: Path | None = None) -> _MemoryStore | _FileStore:
    global _warned_fallback  # noqa: PLW0603 - one warning per process
    try:
        if directory is None:
            from app.services.resource_governor.shared_slots import (
                shared_slots_directory,
            )

            directory = shared_slots_directory(_STATE_DIRECTORY_NAME)
        if directory is None or fcntl is None:
            return _MemoryStore()
        from app.services.resource_governor.shared_slots import ensure_private_directory

        ensure_private_directory(directory)
        return _FileStore(directory / f"{name}.json")
    except OSError as exc:
        if not _warned_fallback:
            _warned_fallback = True
            logger.warning(
                "NVIDIA NIM pacing falls back to per-process state (%s): the request limit counts per process",
                type(exc).__name__,
            )
        return _MemoryStore()


class NvidiaPacer:
    """Sends for one NVIDIA endpoint and key, paced across the container's processes."""

    def __init__(
        self,
        limits: PacerLimits,
        store: _MemoryStore | _FileStore | None = None,
        *,
        clock: Callable[[], float] = time.time,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        sync_sleep: Callable[[float], None] = time.sleep,
        max_wait_s: float = DEFAULT_MAX_WAIT_S,
    ) -> None:
        super().__init__()
        self.limits = limits
        self._store = store if store is not None else _MemoryStore()
        # Wall time, not monotonic: the processes compare the times they write.
        self._clock = clock
        self._async_sleep = async_sleep
        self._sync_sleep = sync_sleep
        self.max_wait_s = max_wait_s

    def _attempt(self, lane: Lane) -> float:
        return self._store.update(lambda state: _try_send(state, self.limits, lane, self._clock()))

    @staticmethod
    def _pause(wait: float) -> float:
        # Waiters woken together would all ask at once; a little spread keeps them from colliding again.
        return wait + random.uniform(0.0, 0.25)

    async def acquire(self, lane: Lane, *, max_wait_s: float | None = None) -> None:
        """Wait until ``lane`` may send, and count the send. Raises ``PacerBusy`` instead of a longer wait."""
        allowance = self.max_wait_s if max_wait_s is None else max_wait_s
        waited = 0.0
        while True:
            if self._store.blocking:
                wait = await asyncio.to_thread(self._attempt, lane)
            else:
                wait = self._attempt(lane)
            if wait <= 0.0:
                return
            if waited + wait > allowance:
                raise PacerBusy(wait)
            pause = self._pause(wait)
            await self._async_sleep(pause)
            waited += pause

    def acquire_sync(self, lane: Lane, *, max_wait_s: float | None = None) -> None:
        allowance = self.max_wait_s if max_wait_s is None else max_wait_s
        waited = 0.0
        while True:
            wait = self._attempt(lane)
            if wait <= 0.0:
                return
            if waited + wait > allowance:
                raise PacerBusy(wait)
            pause = self._pause(wait)
            self._sync_sleep(pause)
            waited += pause

    def penalize(self, cooldown_s: float) -> None:
        """Hold every lane for ``cooldown_s`` after the provider answered 429."""
        clamped = max(0.0, min(float(cooldown_s), _MAX_COOLDOWN_S))
        if clamped <= 0.0:
            return

        def apply(state: _State) -> None:
            state.cooldown_until = max(state.cooldown_until, self._clock() + clamped)

        self._store.update(apply)


_PACERS_LOCK = threading.Lock()
_PACERS: dict[tuple[str, PacerLimits], NvidiaPacer] = {}


def get_nvidia_pacer(base_url: str, api_key: str | None, limits: PacerLimits) -> NvidiaPacer:
    """This process's pacer for an endpoint and key; its state is shared with the container's other processes."""
    # The file name is a digest of the endpoint and key: the same in every process, and it names neither.
    name = hashlib.sha256(f"{base_url}\n{api_key or ''}".encode()).hexdigest()[:32]
    with _PACERS_LOCK:
        pacer = _PACERS.get((name, limits))
        if pacer is None:
            pacer = NvidiaPacer(limits, _open_store(name) if limits.enabled else _MemoryStore())
            _PACERS[(name, limits)] = pacer
        return pacer


def clear_nvidia_pacers() -> None:
    """Forget this process's pacers (tests)."""
    with _PACERS_LOCK:
        _PACERS.clear()
