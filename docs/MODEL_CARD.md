# Model card — experimental v0.1

- Intended use: research and prototyping of media quality checks.
- Not established: production reliability, device-level latency, independent real-world accuracy.
- Source release: code, configs, synthetic smoke generator and tests only; no pretrained weights or datasets.
- Quality metrics: validation is used for selection/tuning. An untouched test set and fixed thresholds are required for final reporting.
- Reproducibility: retain the exact config, seed, source grouping, data version and checkpoint together. GPU and cross-platform bitwise reproducibility are not guaranteed.

## Legacy artifacts

The existing local audio checkpoint is epoch 41 with frame macro F1 about
0.859 on synthetic validation data. It uses legacy clip normalization and is
supported only by full-recording offline inference. New streaming models use
fixed affine normalization and must be trained from scratch. Synthetic anomaly
performance does not establish accuracy on real player failures. Real-source
mode still uses synthetic labels; collect independent labeled recordings to
measure deployment quality.
