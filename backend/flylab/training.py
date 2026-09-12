"""REINFORCE on a synthetic odor-conditioning contextual bandit.

This trains the neural circuit's MB->DN parameters, not a separate controller.
The reward concerns approach/avoid decisions, not learned whole-body walking.
"""
from __future__ import annotations

import copy
import threading
import time
from pathlib import Path
import torch
from .neural import ReferenceBrain


def evaluate(brain, reward_odor="A"):
    with torch.no_grad():
        obs = torch.tensor([[.25, 1., 0., 0.], [.25, 0., 1., 0.]])
        probs = brain.logits(brain.settle(obs)).softmax(-1)
        target = torch.tensor([0, 1] if reward_odor == "A" else [1, 0])
        return {"approach_A": float(probs[0, 0]), "approach_B": float(probs[1, 0]), "expected_reward": float((2*probs[torch.arange(2), target]-1).mean()), "accuracy": float((probs.argmax(-1) == target).float().mean())}


class Trainer:
    def __init__(self, directory: Path):
        self.directory = directory
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.result: ReferenceBrain | None = None
        self.status = {"running": False, "episodes": 0, "total": 0, "history": [], "before": None, "after": None, "error": None, "checkpoint": None, "reward_odor": "A", "parameter_change": 0., "applied": False}

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.status)

    def start(self, brain, episodes=100, reward_odor="A", seed=42):
        with self.lock:
            if self.status["running"]:
                raise ValueError("Training is already running")
            candidate = copy.deepcopy(brain)
            self.stop_event.clear()
            self.result = None
            self.status = {"running": True, "episodes": 0, "total": episodes, "history": [], "before": evaluate(candidate, reward_odor), "after": None, "error": None, "checkpoint": None, "reward_odor": reward_odor, "parameter_change": 0., "applied": False}
            self.thread = threading.Thread(target=self._run, args=(candidate, episodes, reward_odor, seed), daemon=True)
            self.thread.start()

    def _run(self, brain, episodes, reward_odor, seed):
        try:
            generator = torch.Generator().manual_seed(seed)
            optimizer = torch.optim.Adam([brain.plastic], lr=.012)
            initial = brain.plastic.detach().clone()
            for epoch in range(episodes):
                if self.stop_event.is_set():
                    break
                cues = torch.randint(0, 2, (32,), generator=generator)
                obs = torch.zeros(32, 4)
                obs[:, 0] = .25
                obs[torch.arange(32), cues+1] = .65 + .35*torch.rand(32, generator=generator)
                logits = brain.logits(brain.settle(obs))
                probs = logits.softmax(-1)
                action = torch.multinomial(probs.detach(), 1, generator=generator).squeeze(-1)
                target = cues if reward_odor == "A" else 1-cues
                reward = torch.where(action == target, 1., -1.)
                logp = logits.log_softmax(-1)[torch.arange(32), action]
                entropy = -(probs * logits.log_softmax(-1)).sum(-1).mean()
                loss = -(logp * reward).mean() - .01*entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.)
                optimizer.step()
                with torch.no_grad():
                    brain.plastic.clamp_(-1., 1.)
                with self.lock:
                    self.status["episodes"] = epoch+1
                    self.status["history"].append({"episode": epoch+1, "reward": float(reward.mean()), "loss": float(loss.detach())})
                time.sleep(.015)  # cooperative UI/cancellation; not simulation time
            self.directory.mkdir(parents=True, exist_ok=True)
            name = f"odor-{time.time_ns()}.pt"
            path = self.directory/name
            torch.save({"state_dict": brain.state_dict(), "task": "odor_conditioning", "reward_odor": reward_odor, "seed": seed, "model": "synthetic-reference-v1"}, path)
            with self.lock:
                self.result = brain
                self.status.update(after=evaluate(brain, reward_odor), checkpoint=name, parameter_change=float(torch.linalg.norm(brain.plastic.detach()-initial)))
        except Exception as exc:
            with self.lock:
                self.status["error"] = str(exc)
        finally:
            with self.lock:
                self.status["running"] = False

    def apply(self, simulation):
        with self.lock:
            if self.status["running"] or self.result is None:
                raise ValueError("Finish or stop a training run before applying its brain")
            simulation.brain.load_state_dict(self.result.state_dict())
            simulation.reset()
            self.status["applied"] = True

    def load(self, name, simulation):
        # Only local checkpoint basenames, and tensors loaded with safe weights_only.
        if Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("Invalid checkpoint name")
        if self.snapshot()["running"]:
            raise ValueError("Stop training before loading a checkpoint")
        checkpoint = torch.load(self.directory/name, map_location="cpu", weights_only=True)
        if checkpoint.get("model") != "synthetic-reference-v1":
            raise ValueError("Checkpoint belongs to a different model")
        simulation.brain.load_state_dict(checkpoint["state_dict"])
        simulation.reset()

    def checkpoints(self):
        return [p.name for p in sorted(self.directory.glob("*.pt"), reverse=True)] if self.directory.exists() else []
