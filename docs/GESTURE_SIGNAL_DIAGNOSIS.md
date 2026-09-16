# Short-trial diagnostic before full-history training

This report records the earlier 20 ms truncated-gradient implementation. Current training uses full-sample checkpointed gradients and 500 ms by default; see [Gesture learning](GESTURE_LEARNING.md). Rerunning the diagnostic script uses the current algorithm, so its gradient values will differ from this historical record.

The inspected run completed all ten Adam updates, but its best validation MSE improved by only 0.23%. Its five commands cover 100 ms. A full-connectome replay at the original adapter, with the run's exact mechanics, sensory settings, and GPU float16 execution, found no motor-output difference within that period between pointing hands, open palms, randomized pointing-hand placement, or zero retinal drive. The input images and photoreceptor firing did differ. The visual pathway is not missing its input; the training interval ends before measurable motor differentiation.

Extending the diagnostic to 500 ms produced the first measured motor-rate difference at 360 ms, sampled every 20 ms. This is a configuration-specific observed latency, not a universal biological response time. It does not establish that the resulting movement is correct. The long diagnostic holds the final short target only for scoring; it does not certify a longer physical target trajectory.

The inspected backward pass truncated its history every 20 ms. It cannot propagate a later loss back through the complete preceding visual-to-motor history. The five-step gradient remains nonzero with vision removed: gradient norms were 5.20222 with vision and 5.20216 without. The maximum component difference was 0.00012054. The gradient is therefore mostly responding to the existing nonvisual activity in this trial, rather than evidence of learned gesture control.

Finite directional probes also showed poor alignment with measured forward loss. Starting MSE was 0.01340233. Moving 0.1 units along the normalized negative surrogate gradient produced 0.01343602, while moving the same distance in the positive direction produced 0.01330218. A smaller negative step of 0.01 did improve slightly. This is evidence of unreliable local guidance at the tested scales, not a proof that flipping all gradients is correct. Discrete spikes, FP16 rounding, and truncated surrogate differentiation limit the interpretation of smooth-gradient tests.

## Consequences for training

Increasing the number of 100 ms trials cannot be assumed to teach the visual response: these measured trials have identical motor outputs even without vision. Removing early stopping does not address this failure. A useful next training design must allow a measured visual response interval, supervise a physically validated muscle trajectory at that interval, and validate credit assignment across the relevant neural history. Simply increasing the current Steps field is not a verified solution: longer target holds must also be physically checked, and the 20 ms backward truncation remains.

No neural topology, signs, precision, target values, or production training settings were changed by this investigation. Existing saved weights and live state were left intact.

The numerical evidence is in [gesture-signal-diagnosis.json](results/gesture-signal-diagnosis.json). Run an updated diagnostic with:

```sh
PYTHONPATH=backend .venv/bin/python scripts/diagnose_gesture_signal.py \
  --checkpoint data/gesture-checkpoints/gesture-1789409003024982000.pt \
  --output /private/tmp/gesture-signal-diagnosis.json
```

The script uses the saved configuration and starts from the original adapter. It creates an isolated full brain, retains every neural tick, and does not install weights or modify the live experiment.
