"""Profile native integration without changing joints, precision, or timesteps."""
import json
from pathlib import Path
import platform
import time

import mujoco

from flylab.flight import FlightRig, TRIM_FREQUENCY, average_wingbeats


def main():
    rig = FlightRig()
    original = mujoco.get_mjcb_time()
    try:
        mujoco.set_mjcb_time(time.perf_counter)
        started = time.perf_counter()
        samples = rig.flap(.08)
        wall = time.perf_counter()-started
    finally:
        mujoco.set_mjcb_time(original)
    timers = {name: dict(seconds=stat.duration, calls=stat.number)
              for name, stat in zip(mujoco.mjTIMERSTRING, rig.data.timer) if stat.number}
    result = dict(platform=platform.platform(), model=rig.description(), dofs=int(rig.model.nv),
                  simulated_seconds=.08, physics_steps=1600, wall_seconds=wall,
                  timers=timers, means=average_wingbeats(samples, TRIM_FREQUENCY),
                  interpretation='Timers are nested and must not be summed. Includes Python timer callback overhead. '
                  'Advance is the native integration stage, including velocity-dependent force derivatives and solve.')
    output = Path('docs/results/wing-runtime-profile.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('Wall seconds:', wall, '\nNative timers:', timers)


if __name__ == '__main__':
    main()
