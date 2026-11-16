# Validation — 2026-11-16

## Environment

- macOS ARM64, Python 3.14.4, PyTorch 2.12.0, NumPy 2.4.6.
- Dependencies installed into a new venv without system-site-packages.
- Both projects' dev dependencies were installed together; project-specific
  resolved closures are recorded in requirements-lock-macos-py314.txt.
- `pip check`: passed. Main requirements use constraints.txt for direct versions.

## Executed checks

- `python -m pytest -q`: **36 passed**.
- `python scripts/smoke.py`: **passed**, using temporary data/weights/config.
- Source syntax and candidate-file credential-pattern scan completed.
- No datasets, recordings, model weights, virtual environments or IDE settings
  are part of the Git source release.

- Smoke covers synthetic data generation, one CPU training epoch, offline
  inference and error visualization using the generated checkpoint.
- Streaming tests compare complete frame sequences against full causal forward
  at absolute tolerance 2e-6, across short inputs, multiple chunk sizes,
  normalization on/off and causal smoothing. Flush is idempotent.
- Default first emission is within 300 ms of accumulated synthetic input;
  this excludes compute/network delay and is not a hardware latency benchmark.
- Existing manifest: 1,600 training / 400 validation files; **zero path overlap**.
  Legacy manifests without source_id still require recording/session provenance.
- Existing local legacy best.pt offline inference succeeds: 296 frames / 3 s.
- Tests include HTTP concurrency, generated fixtures, source/duplicate grouping,
  overlapping labels, exclusion of padded frames and legacy NumPy thresholds.
- One upstream Starlette warning remains: its TestClient deprecates httpx in
  favor of httpx2. All HTTP tests pass with the constrained environment.

## Not established by this run

- GitHub-hosted CI execution, Linux/Windows runtime or CUDA behavior.
- Independent real-recording quality, full retraining, or production latency.
- Optional labeling UI / ONNX / external dataset download integration.

The historical quality scores described in MODEL_CARD.md remain legacy
validation evidence, not new independent test results. Existing user data and
weights were retained; new smoke outputs were isolated in temporary directories.
