"""One in-flight simulated cycle, with wall-clock pacing and deadline telemetry."""
import asyncio
import math
from collections import deque


class CycleScheduler:
    def __init__(self):
        self.generation = 0
        self.reset()

    def reset(self):
        self.generation += 1
        self.durations = deque(maxlen=60)
        self.budget = 0.
        self.compute = 0.
        self.misses = 0
        self.cycles = 0
        self.phase = 'idle'

    def snapshot(self):
        return {'phase': self.phase, 'target_hz': 1 / self.budget if self.budget else None,
                'achieved_hz': len(self.durations) / sum(self.durations) if self.durations else None,
                'budget_ms': self.budget * 1000, 'compute_ms': self.compute * 1000,
                'deadline_misses': self.misses, 'completed_cycles': self.cycles,
                'over_budget': self.phase == 'over-budget' or self.compute > self.budget > 0}

    async def run(self, work, budget):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('Cycle budget must be finite and positive')
        generation = self.generation
        loop = asyncio.get_running_loop()
        started = loop.time()
        self.budget, self.phase = budget, 'computing'
        task = asyncio.create_task(work())
        try:
            done, _ = await asyncio.wait({task}, timeout=budget)
            if not done and generation == self.generation:
                # A deadline is not a cancellation point: the worker must finish
                # the atomic brain/body step before any other step can begin.
                self.phase = 'over-budget'
            result = await asyncio.shield(task)
            duration = loop.time() - started
            if generation == self.generation:
                self.compute = duration
                self.misses += int(duration > budget)
                self.phase = 'waiting' if duration < budget else 'over-budget'
            await asyncio.sleep(max(0., budget - (loop.time() - started)))
            if generation == self.generation:
                self.durations.append(loop.time() - started)
                self.cycles += 1
                self.phase = 'over-budget' if duration > budget else 'on-time'
            return result
        except asyncio.CancelledError:
            # Shutdown still waits for a running worker; no detached simulation.
            await task
            raise
