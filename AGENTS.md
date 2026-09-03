# FrayID public contributor guide

Before making changes, read these files completely and in order:

1. `docs/PROJECT_CONTEXT.md`;
2. `docs/HANDOFF.md`;
3. `docs/AI_REVIEW_GUIDE.md`;
4. `configs/evaluation/post_v1_experiments.yaml`;
5. `configs/evaluation/current_project_track.yaml`;
6. `docs/ROADMAP.md`;
7. `PRIVACY.md`.

Treat the public handoff as the factual record and the roadmap as unaccepted
candidate work. Do not infer missing private artifacts from filenames or add
private provenance to the public repository.

## Goal

First reproduce a SelfRecon-style unified canonical outer human surface from
sequence-level evidence in a cooperative monocular self-rotation video. The
primary representation is an implicit SDF with skeletal and bounded residual
non-rigid deformation. Explicit open-garment topology and cloth physics are
deferred research, not prerequisites for this result.

## Boundaries

- V1 is geometry-only: do not add measurement, garment, avatar, texture, 3DGS,
  or virtual-try-on scope.
- Reproduce the official/reference outer-surface path before adding a frontier
  garment representation. Garment boundaries may be diagnostics but are not
  required primary state.
- CameraHMR is initialization, Sapiens2 is evidence, SMPL is a scaffold, and
  the SDF is the canonical surface representation.
- Never substitute proxy masks, constant normals, proxy cameras, or zero poses
  for missing trained output.
- Never declare success because an output file exists; use held-out geometric
  metrics and topology checks.
- A bare-torso capture may validate camera/pose/SDF plumbing only; never call it
  clothed-human or garment reconstruction.
- Never commit human media or derived body data, local manifests, weights,
  checkpoints, meshes, outputs, credentials, or cloud identifiers.
- Do not propose making the private canonical repository public. Its reachable
  history is outside this fresh-history publication boundary.

## Verification

```bash
make verify
```
