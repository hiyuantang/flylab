"""Download official versioned inputs atomically, then use the separate importer."""
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backend'))
from flylab.connectome import FILES, SOURCE
root=Path(__file__).resolve().parents[1]/'data'
root.mkdir(exist_ok=True)
for local,remote in FILES.items():
    destination=root/local
    if destination.exists():
        print(f'Keeping existing {local}; delete explicitly to download again.')
        continue
    temporary=destination.with_suffix('.part')
    print(f'Downloading {remote}', flush=True)
    urllib.request.urlretrieve(SOURCE+remote, temporary)
    temporary.replace(destination)
print('Run: PYTHONPATH=backend uv run python -m flylab.connectome')
