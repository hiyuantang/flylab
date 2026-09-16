"""Evaluate a saved visual policy on fresh closed-loop physical episodes."""
import argparse
import hashlib
import json
from pathlib import Path
from threading import Event

from flylab.policy_evaluation import evaluate_policy
from flylab.policy_training import PolicyTrainer, policy_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Policy filename in the checkpoint directory')
    parser.add_argument('--directory', type=Path, default=Path('data/policy-checkpoints'))
    parser.add_argument('--start-seed', type=int, required=True, help='Use seeds excluded from training and selection')
    parser.add_argument('--episodes-per-gesture', type=int, default=8)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.episodes_per_gesture < 1 or not 5 <= args.steps <= 250 or args.start_seed < 0:
        parser.error('Require positive episode count, nonnegative seed, and 5–250 steps')
    trainer = PolicyTrainer(args.directory)
    saved = trainer._load(args.checkpoint)
    simulation = trainer.load_simulation(args.checkpoint)
    cues = [cue for cue, weight in saved['metadata']['proportions'].items() if weight > 0]
    result = evaluate_policy(simulation.policy, saved['settings'], cues,
        seeds=range(args.start_seed, args.start_seed + args.episodes_per_gesture),
        horizon=args.steps, stop=Event())
    payload = dict(checkpoint=args.checkpoint,
        checkpoint_sha256=hashlib.sha256((args.directory / args.checkpoint).read_bytes()).hexdigest(),
        device=policy_device(), precision='float32', evaluation=result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(f"{result['successes']}/{result['episodes']} physical trials passed; {args.output}")


if __name__ == '__main__':
    main()
