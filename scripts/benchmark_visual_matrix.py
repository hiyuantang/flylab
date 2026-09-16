"""Compare compound-eye quadrature and receptor matrix processing on CPU/MPS.

Ray intersections remain MuJoCo CPU work. Includes CPU->GPU radiance transfer and
GPU->CPU receptor readback, because these are required by the current sensor API.
"""
from pathlib import Path
import json
import statistics
import time
import numpy as np
import torch
from flylab.retina import integrate_facets, receptor_channels


def main():
    rng = np.random.default_rng(42)
    rgb = rng.random(((857 + 852) * 7, 3))
    def cpu():
        channels = receptor_channels(integrate_facets(rgb))
        return np.stack([channels[k] for k in ['R1-R6', 'R8p', 'R8y']], axis=1)
    quadrature = torch.tensor([.4, .1, .1, .1, .1, .1, .1], device='mps')
    receptors = torch.tensor([[.05, 0, 0], [.65, 0, 1], [.30, 1, 0]], device='mps')
    def gpu():
        samples = torch.as_tensor(rgb, dtype=torch.float32, device='mps').reshape(-1, 7, 3)
        facets = (quadrature @ samples).clamp(0, 1)
        return (facets @ receptors).cpu().numpy()
    expected, actual = cpu(), gpu()
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
    trials = []
    for trial in range(5):
        for mode, fn in ([('cpu', cpu), ('gpu_matrix', gpu)] if trial % 2 == 0 else [('gpu_matrix', gpu), ('cpu', cpu)]):
            fn(); torch.mps.synchronize()
            start = time.perf_counter()
            for _ in range(100): fn()
            torch.mps.synchronize()
            trials.append(dict(trial=trial, mode=mode,mean_ms=10*(time.perf_counter()-start)))
    result = dict(scope=__doc__,max_abs_error=float(np.max(np.abs(actual-expected))),trials=trials,
                  median_ms={m:statistics.median(t['mean_ms'] for t in trials if t['mode']==m) for m in ['cpu','gpu_matrix']})
    Path('docs/results/visual-matrix.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))


if __name__ == '__main__': main()
