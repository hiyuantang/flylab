"""Physical policy rollouts, separate from supervised action prediction error."""
import numpy as np

from .gesture_scene import HandStimulus, random_placement
from .gesture_targets import evaluate_reference
from .policy_simulation import PolicySimulation


def evaluate_policy(model, settings, cues, *, seeds, horizon, stop, publish=None):
    """Every held-out episode uses the policy's own vision and body trajectory.

    A successful episode must satisfy the existing physical pose and support
    criteria for all of its last five command steps. Labels are evaluator-only.
    """
    episodes = []
    for cue in cues:
        for seed in seeds:
            if stop.is_set():
                from .gesture_training import Cancelled
                raise Cancelled()
            sim = PolicySimulation(model, settings)
            initial_height = float(sim.body.data.qpos[2])
            sim.sensors.visual_object = HandStimulus(cue,
                placement=random_placement(np.random.default_rng(seed)))
            metrics = []
            for _ in range(horizon):
                if stop.is_set():
                    from .gesture_training import Cancelled
                    raise Cancelled()
                sim.advance()
                metrics.append(evaluate_reference(sim.body, cue, initial_height))
            episodes.append(dict(gesture=cue, seed=int(seed),
                success=all(m['pose_reached'] for m in metrics[-5:]),
                successful_frames=sum(m['pose_reached'] for m in metrics),
                frames=horizon, final=metrics[-1]))
            if publish:
                publish(cue, sim, len(episodes))
    by_gesture = {}
    for cue in cues:
        rows = [row for row in episodes if row['gesture'] == cue]
        by_gesture[cue] = dict(successes=sum(row['success'] for row in rows), episodes=len(rows))
    successes = sum(row['success'] for row in episodes)
    return dict(success_rate=successes / len(episodes), successes=successes,
                episodes=len(episodes), by_gesture=by_gesture, trials=episodes,
                criterion='Pose and support criteria held for the final five 20 ms steps',
                simulated_seconds_per_episode=horizon * .02)
