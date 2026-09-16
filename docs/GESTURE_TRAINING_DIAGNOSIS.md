# Full-history training diagnosis

The stopped 500 ms run accepted all six completed Adam updates. Its baseline validation MSE was approximately 0.0168221; the best completed value was 0.0167311 (about 0.54% lower), and the sixth update returned to 0.0168237. The optimizer was running, but these results do not demonstrate the requested pose.

The unclipped surrogate-gradient norm ranged from approximately 10^177 to 10^200. Binary rescaling prevents overflow and global norm clipping bounds the optimizer input; neither establishes that the surrogate direction improves the discrete forward model.

An isolated full-connectome diagnostic used the run's configuration, original adapter and physically verified 500 ms target. Its baseline MSE was 0.01682211. Moving 0.001 units along the normalized negative surrogate gradient worsened MSE to 0.01687027. A negative displacement of 0.01 improved it to 0.01669918, whereas a positive displacement of 0.1 improved it to 0.01661084. This irregular response across scales is evidence of unreliable local guidance, not justification for reversing every update.

About 80.60% of squared parameter-gradient magnitude lay in the receiving ascending-neuron group and 19.36% in the broad VNC-intrinsic group. The dominant recurrent backward signal leaves comparatively little gradient for other groups. This supports recurrent surrogate-gradient amplification as a major training limitation. The small dense reference tests verify the implemented surrogate equations, not their usefulness for learning the hard-spiking physical model.

Vision is not completely disconnected at 500 ms: removing retinal drive changed final muscle activations by up to 0.02196, and replacing pointing hands with palms changed them by up to 0.01316. However, most required muscle activations remain far below their targets. In the stopped run's last published sample, only 19 channels exceeded 0.0001 activation versus 52 in the target. These observations do not establish that the constrained adapter can reach every target.

The next training-method investigation should calibrate the surrogate derivative and test its gradient direction and repeated learning on fixed inputs before recommending more long runs. Simply increasing duration, disabling validation, or clipping the final gradient does not resolve the demonstrated instability. No change to the neural forward model or derivative was made by this diagnostic.

## Preview fix and validation

The GPU preview now publishes after forward simulation, before backpropagation and validation. Each backward checkpoint publishes progress and elapsed time. The completed run history and saved weight files remain separate from this pre-update response preview.

Thirty existing batch/parallel tests and one new ordering/progress test passed on Apple GPU. The new test asserts that the frame exists before the backward routine starts and checks all checkpoint progress events. The frontend production build passed. Chrome at localhost:8000 was checked after the authorized restart; it showed the stopped/discarded run and remained connected. A fresh full-size training run was not launched in the user's controller for UI testing.

Evidence: [isolated gradient probes](results/gesture-long-history-diagnosis.json).

Reproduce without changing live or saved weights:

```sh
PYTHONPATH=backend .venv/bin/python scripts/diagnose_full_history.py \
  --checkpoint data/gesture-checkpoints/gesture-1789409003024982000.pt \
  --output /private/tmp/full-history-diagnosis.json
```
