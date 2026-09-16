"""Bounded brain/body overlap with a barrier at each simulated-time boundary."""
from concurrent.futures import ThreadPoolExecutor

# Only physical work runs here. The caller owns brain state; its workbench lock
# remains held until both branches finish. Workers are joined at process exit.
_PHYSICS = ThreadPoolExecutor(max_workers=2, thread_name_prefix='flylab-physics')
COUPLING_VERSION = 'previous-command-v1'


def run_pair(brain_work, body_work, *, concurrent=True):
    """Return brain output only after both branches finish, including on failure.

    concurrent=False is the deterministic delayed reference for validation. It
    uses the same inputs and equations, differing only in wall-clock scheduling.
    """
    if not concurrent:
        result = brain_work()
        body_work()
        return result
    physical = _PHYSICS.submit(body_work)
    try:
        return brain_work()
    finally:
        # Never leave a background mutation behind after a timeout/exception.
        physical.result()
