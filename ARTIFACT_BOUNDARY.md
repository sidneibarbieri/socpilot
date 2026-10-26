# Release Boundary

This package is the public artifact boundary for reproducibility. It is the
software and data package used to validate and rerun SOCpilot experiments. It is
not a publication workspace and does not include literature reviews, private
notes, or internal planning material.

- Contains only anonymized inputs under `artifact_data/`: two disjoint export
  batches in `dataset/` and `heldout/`, plus global policy inputs in `global/`.
- Excludes raw incoming data and secret-bearing local environment files.
- Includes the deterministic anonymization and conversion code under `local_redaction/`.
- Excludes local secrets (`.env`) and developer caches.
- Excludes publication workspaces, literature folders, private notes, and other
  author-only administrative resources.
- Includes optional analysis outputs under `artifact_outputs/` when available.
- The terminal evidence manifest, when present, is the sole authority for model
  evidence and binds every study to one protocol freeze.
- Includes the exact per-task model inputs, model outputs, verifier records, provider-transport metadata, diagnostic ledgers, and aggregate reports bound by the terminal evidence manifest.
- Excludes credentials, authorization headers, private raw exports, and private incident lineage.
