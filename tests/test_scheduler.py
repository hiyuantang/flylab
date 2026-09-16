import asyncio
import time
import pytest
from flylab.scheduler import CycleScheduler


def test_early_result_waits_until_deadline():
    async def check():
        scheduler = CycleScheduler()
        async def work():
            return 'completed'
        started = time.monotonic()
        task = asyncio.create_task(scheduler.run(work, .04))
        await asyncio.sleep(.005)
        assert not task.done()
        assert scheduler.snapshot()['phase'] == 'waiting'
        assert await task == 'completed'
        assert time.monotonic() - started >= .035
        assert scheduler.snapshot()['completed_cycles'] == 1
        assert scheduler.snapshot()['deadline_misses'] == 0
    asyncio.run(check())


def test_late_work_is_not_cancelled_or_duplicated():
    async def check():
        scheduler = CycleScheduler()
        release = asyncio.Event()
        count = 0
        async def work():
            nonlocal count
            count += 1
            await release.wait()
            return 'full-step'
        task = asyncio.create_task(scheduler.run(work, .005))
        await asyncio.sleep(.02)
        assert not task.done() and scheduler.snapshot()['over_budget']
        release.set()
        assert await task == 'full-step' and count == 1
        assert scheduler.snapshot()['deadline_misses'] == 1
        assert scheduler.snapshot()['compute_ms'] >= 5
    asyncio.run(check())


def test_cancellation_waits_for_inflight_worker():
    async def check():
        scheduler = CycleScheduler()
        release, started = asyncio.Event(), asyncio.Event()
        completed = []
        async def work():
            started.set()
            await release.wait()
            completed.append(True)
        task = asyncio.create_task(scheduler.run(work, .1))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed == [True]
    asyncio.run(check())
