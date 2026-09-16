# Backward surrogate calibration

New training runs use `surrogate_scale=0.0003`. This is a training hyperparameter, not a measured biological constant. The spike surrogate derivative is

`scale / (1 + abs(voltage_mV + 45))²`.

The hard forward threshold, reset, refractory decisions, precision, every imported neuron/connection, and every 0.1 ms tick are unchanged. There are no new skip connections. Full-sample BPTT and its checkpoint boundaries remain intact. CPU and Apple GPU implementations apply the same scale, and parallel GPU samples share the configured kernel. The scale is recorded in run status, checkpoint settings, and version metadata; the training API accepts an explicit override in `(0, 1]`. Resuming a paused run preserves its existing engine and scale. Starting from an old adapter uses the new requested/default training scale without changing its inference compatibility.

## Isolated full-connectome comparison

The comparison starts from the original adapter, with rank 2, the existing FP16 forward configuration, a physically checked 500 ms two-front-leg target, and fixed hand placement. Every candidate had the same initial forward MSE: 0.01682211168. Each one-update Adam trial starts fresh; no trial weights are installed in the live controller.

| Derivative scale | log10 of raw gradient norm | MSE after one Adam update, learning rate 0.01 |
| --- | ---: | ---: |
| 1 | 187.30 | 0.01692703 |
| 0.1 | 76.59 | 0.01679959 |
| 0.03 | 36.66 | 0.01669819 |
| 0.01 | 11.07 | 0.01683987 |
| 0.003 | -4.61 | 0.01682757 |
| 0.001 | -5.12 | 0.01678414 |
| 0.0003 | -5.66 | 0.01674856 |

The larger candidates still amplify gradients severely despite occasional loss improvements. The two stable candidates that improved loss at learning rate 0.01 were checked over three Adam updates, with randomized training placement and a fixed validation placement. Scale 0.001 ended at 0.01682791, slightly worse than baseline. Scale 0.0003 ended at 0.01670784, about 0.68% better than baseline. Its raw gradient norm stayed between 1.97e-6 and 2.26e-6. This completed comparison supports choosing 0.0003 as the current default, not as a universally optimal value.

The selected scale reduced concentration in one receiving-neuron group from about 80.6% to 25.0% of squared parameter-gradient magnitude in the initial trial. It addresses the demonstrated explosion; it does not establish reliable long-range credit assignment. A smaller surrogate can attenuate long paths. The hard-spike approximation, detached physical/sensory feedback, constrained adapter, and sparse motor activity remain limitations. Only one gesture and one starting configuration were used for calibration. Generalization and the requested leg-raising behavior are not demonstrated by this small loss improvement.

## Validation and reproduction

Dense PyTorch checks cover the scaled backward equations with zero and 1.8 ms delays, full-history checkpoint replay, and parallel/sequential equivalence. Exact forward-state comparisons against the existing GPU inference path pass at the selected scale. The forward compatibility digest remains `57f7d3490183f2ec42ee48e39f4ae1b9f3b3835577b9581bbc3207394637d9ba`. API validation and checkpoint metadata tests also pass.

Evidence:

- [Initial scale sweep](results/surrogate-scale-calibration.json)
- [Small-scale sweep](results/surrogate-scale-calibration-small.json)
- [Three updates at 0.001](results/surrogate-stable-learning.json)
- [Three updates at 0.0003](results/surrogate-stable-learning-small.json)

Run `scripts/calibrate_surrogate_scale.py` or `scripts/check_surrogate_learning.py` with `--checkpoint` for the configuration source and `--output` for the report path. The latter accepts `--scale`. Both create isolated model instances and leave existing checkpoints and the live controller untouched.

The scale investigation follows the finding that surrogate scale strongly affects recurrent SNN learning: [Zenke and Vogels, 2021](https://direct.mit.edu/neco/article/33/4/899/97482/The-Remarkable-Robustness-of-Surrogate-Gradient). The numerical choice here comes from the local experiments above, not from copying a value from that paper.
