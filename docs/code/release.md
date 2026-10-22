# Release scope and provenance

This is an **initial motion inference/evaluation source release**. It is not
yet the complete training or benchmark reproduction package.

## Included

- The public upstream main motion model (`model/sif3d_final.py`) and its
  attention, PointNet2, GCN, and SMPL-X conversion dependencies.
- The GIMO-format evaluation dataset loader.
- `predict.py` for observed-input-only motion prediction, and `evaluate.py`
  for the four future-motion metrics.
- Setup, input/output contracts, compatibility checks, and CPU unit tests.

## To be released

- Pretrained motion and interaction-prior checkpoints.
- Confirmed teacher/student training code and final configurations.
- Interaction-prior generation, raw-data preprocessing, and benchmark-specific
  reproduction instructions.

The upstream repository retains many historical training, distillation, and
selector variants and explicitly does not identify the paper's final
configuration. Those variants, experimental launchers, additional dataset
experiments, ablations, and generated results are deferred. No claim that this
source package reproduces the reported paper scores is made.

## Source

Code was selected from
[Yoon-Ji-Il/NeurIPS2026-SGIP](https://github.com/Yoon-Ji-Il/NeurIPS2026-SGIP)
at commit
[`823c019`](https://github.com/Yoon-Ji-Il/NeurIPS2026-SGIP/commit/823c019ce7bd48449ba02bcb2b0ce0d98e03100a).
The [source manifest](source_manifest.json) maps imported/adapted files to their
original paths and records their SHA-256 hashes.

Packaging changes:

- Descriptive module paths and an `SGIP` alias for the upstream `SIF3D` class;
  learned parameter names and core model computations are preserved.
- An inference-focused configuration for the supported 6/10-frame protocol.
- Input validation and dataset row/shape checks.
- A shared body-model/checkpoint loader and future-only prediction export.
- Equivalent evaluation errors reported as JSON, with sample-weighted
  aggregation and no dropped final batch.
- Unused backbone variants, decoder helpers, and standalone demos are omitted.
- Heavy model dependencies are loaded only when inference begins.

## Validation limits

CPU checks exercise input contracts, partial-batch inference with a stand-in
model, checkpoint compatibility, metric formulas, and synthetic cache loading.
The core forward path is also checked with lightweight stand-ins for PointNet2
and the external body models; this does not validate those external components.
During packaging, the retained backbone/attention/GCN definitions were checked
against the upstream source, and the core outputs matched exactly under
identical weights, inputs, and external stand-ins.

Full CUDA inference and the paper's benchmark results have not been rerun:
compatible research checkpoints, prepared data, and external body-model
assets are not bundled. The dependency list is not an exact reproduction lock.
See [third-party notices](../../THIRD_PARTY_NOTICES.md) for attribution.
