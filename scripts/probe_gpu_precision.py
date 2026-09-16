"""Probe local MPS arithmetic support without touching the live simulation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import torch


def probe():
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'torch': torch.__version__, 'mps_available': torch.backends.mps.is_available(),
              'probes': {}}
    for name in ['float16', 'float8_e4m3fn', 'float8_e5m2']:
        row = {'allocation': False, 'multiply': False}
        result['probes'][name] = row
        try:
            x = torch.tensor([1., 2.], dtype=getattr(torch, name), device='mps')
            row['allocation'] = True
            y = x * x
            torch.mps.synchronize()
            row.update(multiply=True, values=y.cpu().float().tolist())
        except (TypeError, RuntimeError, NotImplementedError) as exc:
            row['error'] = str(exc)
    return result


if __name__ == '__main__':
    result = probe()
    output = Path('docs/results/gpu-low-precision-support.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
