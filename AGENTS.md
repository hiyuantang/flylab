# Repository Guidelines

## Project Structure & Module Organization

- `backend/flylab/`: PyTorch circuits, MuJoCo body mechanics, simulation orchestration, training, connectome import, and FastAPI endpoints. Keep these responsibilities in their existing modules.
- `frontend/src/components/`: React/Three.js views; `frontend/src/lib/`: shared API contracts and client helpers.
- `tests/`: Python behavioral tests. `scripts/`: local launcher and dataset downloader.
- `frontend/public/`: licensed meshes and asset notices. `docs/`: design, provenance, screenshots, and validation records. `data/`: ignored datasets and checkpoints.

## Build, Test, and Development Commands

Run from the repository root unless noted. Use Python 3.12 and Node.js 22.

- `uv sync --python 3.12`: install locked Python dependencies.
- `(cd frontend && npm ci)`: install locked frontend dependencies.
- `./scripts/dev.sh`: launch FastAPI on port 8000 and Vite on 5178; installs missing dependencies.
- `PYTHONPATH=backend uv run pytest -q`: run backend tests.
- `(cd frontend && npm run build)`: type-check TypeScript and produce `frontend/dist/`, including dependency license notices.

## Coding Style & Naming Conventions

Use custom dropdown menus throughout the UI; never use native HTML `<select>` controls. Reuse `frontend/src/components/Select.tsx` for selection menus.

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Use two-space TypeScript indentation, `camelCase` functions/variables, and `PascalCase.tsx` component files. Preserve strict TypeScript types and update shared contracts when API payloads change.

Format changed frontend files with the installed Prettier, for example, from `frontend/`: `npx prettier --write src/components/Brain.tsx`. No Python formatter or lint command is configured.

## Testing Guidelines

Use pytest with `tests/test_*.py` files and `test_*` functions. Cover changed behavior with deterministic seeds and temporary checkpoint paths. Prioritize neural interventions, physical effects, learning isolation, checkpoint round trips, and API validation.

The measured-data integrity test skips when bulk data is absent. No numeric coverage threshold or frontend test runner is configured; verify changed UI interactions in the browser alongside the production build.

## Commit & Pull Request Guidelines

Use short imperative summaries, following `Initialize FlyLab simulation workbench`. No Conventional Commits requirement exists. PRs should explain the problem, resulting behavior, and validation; link related issues and include screenshots for visible changes.

## Scientific Integrity & Contribution Boundaries

Preserve every imported neuron and connection. Optimize execution without pruning, reduced circuits, skipped neural steps, or silent precision changes; accept slow motion and report simulated versus wall time. Validate dynamics against the scientific reference before claiming fidelity.

Distinguish the synthetic embodied circuit from the measured connectome probe. Document assumptions and preserve source counts, provenance, and third-party notices. Keep credentials, datasets, checkpoints, and build outputs out of Git. Bind the shared local controller to loopback.

Agents own integration and final review. Delegate only repetitive, independent batches to `luna_worker`, with exclusive ownership; verify results and preserve others' edits. Keep architecture, accuracy reviews, security, and external changes with the main agent.
