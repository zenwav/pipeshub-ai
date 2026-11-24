"""The NVIDIA NIM pacer: lanes, the 60 s window, 429 cooldowns, and state shared through a file."""

from __future__ import annotations

import asyncio
import multiprocessing
import sys
from typing import TYPE_CHECKING

import pytest

from app.utils import nvidia_pacer
from app.utils.nvidia_pacer import (
    WINDOW_S,
    Lane,
    NvidiaPacer,
    PacerBusy,
    PacerLimits,
    _FileStore,
    _MemoryStore,
    _open_store,
)

if TYPE_CHECKING:
    from pathlib import Path

posix_only = pytest.mark.skipif(nvidia_pacer.fcntl is None, reason="needs POSIX file locks")


class _Clock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def sleep_sync(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _pacer(limits: PacerLimits, clock: _Clock, **kwargs: float) -> NvidiaPacer:
    return NvidiaPacer(
        limits, _MemoryStore(), clock=clock, async_sleep=clock.sleep, sync_sleep=clock.sleep_sync, **kwargs,
    )


def test_limits_keep_at_least_one_bulk_send_and_zero_disables() -> None:
    assert PacerLimits.of(36, 6) == PacerLimits(36, 30)
    assert PacerLimits.of(4, 10) == PacerLimits(4, 1)
    assert PacerLimits.of(0, 6).enabled is False


async def test_bulk_sends_are_spaced_evenly() -> None:
    clock = _Clock()
    pacer = _pacer(PacerLimits(36, 30), clock)
    for _ in range(3):
        await pacer.acquire(Lane.BULK)
    # 30 a minute: one every 2 s, plus at most 0.25 s of spread per wait.
    assert len(clock.sleeps) == 2
    assert all(2.0 <= s <= 2.25 for s in clock.sleeps)


async def test_a_search_is_not_queued_behind_bulk_sends() -> None:
    clock = _Clock()
    pacer = _pacer(PacerLimits(36, 30), clock)
    await pacer.acquire(Lane.BULK)
    await pacer.acquire(Lane.INTERACTIVE)
    assert clock.sleeps == []


async def test_bulk_stops_at_its_share_and_searches_use_the_rest() -> None:
    clock = _Clock()
    pacer = _pacer(PacerLimits(4, 2), clock, max_wait_s=1_000.0)
    await pacer.acquire(Lane.BULK)
    await pacer.acquire(Lane.BULK)  # two a minute: 30 s apart
    slept = sum(clock.sleeps)
    # Bulk has used its 2 of the window: the next bulk send waits for the window to move.
    with pytest.raises(PacerBusy) as busy:
        await pacer.acquire(Lane.BULK, max_wait_s=5.0)
    assert busy.value.wait_s > 5.0
    # Searches still have the reserve, without waiting.
    await pacer.acquire(Lane.INTERACTIVE)
    await pacer.acquire(Lane.INTERACTIVE)
    assert sum(clock.sleeps) == slept
    # The whole window is now used, by searches too.
    with pytest.raises(PacerBusy):
        await pacer.acquire(Lane.INTERACTIVE, max_wait_s=5.0)
    # Once the window has moved past the first sends, both lanes go again.
    clock.now += WINDOW_S
    await pacer.acquire(Lane.INTERACTIVE, max_wait_s=0.0)


async def test_a_429_cooldown_holds_every_lane() -> None:
    clock = _Clock()
    pacer = _pacer(PacerLimits(36, 30), clock)
    pacer.penalize(15.0)
    await pacer.acquire(Lane.INTERACTIVE)
    assert 15.0 <= clock.sleeps[0] <= 15.25


async def test_a_longer_wait_than_allowed_raises_instead_of_sleeping() -> None:
    clock = _Clock()
    pacer = _pacer(PacerLimits(36, 30), clock, max_wait_s=45.0)
    pacer.penalize(120.0)
    with pytest.raises(PacerBusy) as busy:
        await pacer.acquire(Lane.BULK)
    assert busy.value.wait_s == pytest.approx(120.0)
    assert clock.sleeps == []


async def test_a_cancelled_waiter_leaves_nothing_booked() -> None:
    clock = _Clock()
    released = asyncio.Event()

    async def blocked_sleep(_seconds: float) -> None:
        await released.wait()

    pacer = NvidiaPacer(PacerLimits(36, 30), _MemoryStore(), clock=clock, async_sleep=blocked_sleep)
    await pacer.acquire(Lane.BULK)
    waiter = asyncio.create_task(pacer.acquire(Lane.BULK))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    # The cancelled caller booked nothing: one spacing after the first send, the next goes at once.
    clock.now += 2.0
    await pacer.acquire(Lane.BULK, max_wait_s=0.0)


def test_disabled_pacing_only_honours_a_cooldown() -> None:
    clock = _Clock()
    pacer = _pacer(PacerLimits.of(0, 6), clock)
    for _ in range(100):
        pacer.acquire_sync(Lane.BULK)
    assert clock.sleeps == []
    pacer.penalize(10.0)
    pacer.acquire_sync(Lane.BULK)
    assert 10.0 <= clock.sleeps[0] <= 10.25


def test_stale_state_from_another_run_is_dropped() -> None:
    clock = _Clock()
    store = _MemoryStore()
    store.update(lambda s: setattr(s, "cooldown_until", clock.now + 10 * 3600.0))
    store.update(lambda s: s.sends.extend([clock.now + 5 * 3600.0] * 50))
    pacer = NvidiaPacer(PacerLimits(36, 30), store, clock=clock, async_sleep=clock.sleep, sync_sleep=clock.sleep_sync)
    pacer.acquire_sync(Lane.BULK, max_wait_s=0.0)


def test_unparseable_state_file_content_starts_fresh() -> None:
    from app.utils.nvidia_pacer import _State

    assert _State.load(b"{not json").sends == []
    assert _State.load(b"[1, 2]").cooldown_until == 0.0
    assert _State.load(b'{"sends": ["x", 3, true], "bulk_next": "y"}').sends == [0.0, 0.0, 3.0]


def test_without_a_shared_directory_each_process_keeps_its_own_state(tmp_path: Path) -> None:
    if nvidia_pacer.fcntl is None:
        assert isinstance(_open_store("x"), _MemoryStore)
        return
    plain_file = tmp_path / "not-a-directory"
    plain_file.write_text("")
    assert isinstance(_open_store("x", plain_file), _MemoryStore)


@posix_only
def test_file_store_is_private_and_persists_state(tmp_path: Path) -> None:
    directory = tmp_path / "pacer"
    store = _open_store("abc", directory)
    assert isinstance(store, _FileStore)
    store.update(lambda s: s.sends.append(1.0))
    again = _open_store("abc", directory)
    assert again.update(lambda s: list(s.sends)) == [1.0]
    assert (directory / "abc.json").stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700


@posix_only
def test_a_deleted_state_file_is_opened_again(tmp_path: Path) -> None:
    directory = tmp_path / "pacer"
    store = _open_store("abc", directory)
    store.update(lambda s: s.sends.append(1.0))
    (directory / "abc.json").unlink()
    store.update(lambda s: s.sends.append(2.0))
    # The process that saw the deletion writes to the new file, where later processes look.
    assert _open_store("abc", directory).update(lambda s: list(s.sends)) == [2.0]


def _send_in_child(directory: str, results: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    from pathlib import Path

    from app.utils.nvidia_pacer import (
        Lane,
        NvidiaPacer,
        PacerBusy,
        PacerLimits,
        _open_store,
    )

    pacer = NvidiaPacer(PacerLimits(6, 3), _open_store("shared", Path(directory)), max_wait_s=0.0)
    sent = 0
    for _ in range(10):
        try:
            pacer.acquire_sync(Lane.INTERACTIVE)
            sent += 1
        except PacerBusy:
            pass
    results.put(sent)


@posix_only
@pytest.mark.skipif(sys.platform != "linux", reason="fork start method")
def test_processes_share_one_window(tmp_path: Path) -> None:
    """Four processes asking at once get six sends between them, not six each."""
    directory = tmp_path / "pacer"
    ctx = multiprocessing.get_context("fork")
    results = ctx.Queue()
    workers = [ctx.Process(target=_send_in_child, args=(str(directory), results)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    total = sum(results.get(timeout=5) for _ in workers)
    assert total == 6
