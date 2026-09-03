# AI review guide

This repository is a privacy-safe source and context snapshot for technical
review. It is not a reproduction package for the private subject, and missing
human media or derived geometry must not be inferred, requested, or replaced
with proxies.

## Current decision boundary

V1 is complete and immutable. A separately versioned original-reference
topology correction passed the unchanged development gates without reopening
the sealed test. P0/P2/P5 passed. E1-E17/E19-E21/B1 and
P1/P3/P4/P6-P11 are closed, E18/E24 are blocked, E22/E23/B2/E25 are closed,
and B3 is a closed independent operational baseline. B2 bound its genuine
public inputs. Its first
image build stopped before GPU execution on an interactive system-package
prompt. A second pre-GPU build stopped on historical CMake/Cython build
assumptions. The corrected image passed, but pre-dispatch validation rejected
the conditional training disk request. After that correction, the single L40S
worker failed during Modal bootstrap because the legacy Python 3.8 environment
shadowed the runner's required Python; B2 closed without retry or training.
E11's public
exact-topology gate passed but
its fidelity gate failed on concavity and external-gap preservation. P1's L40S
trace failed the fixed bitwise geometry-gradient gate. P8-P11 then exhausted
the Planar-DAT implementation route: the faithful full singleton filter was
fast and deterministic, but its composed endpoint had 317 exact
self-intersection pairs and its global path oracle timed out. E8 remains
closed.

G16/E16 then exceeded the fixed public cap while constructing the exact fine
ambient scaffold. E17's coarse bi-Lipschitz map retained only `0.01626` of
tangential motion. Exact determinant maps raised retention through `0.07463`
(E19), `0.09314` (E20), and `0.13660` (E21), but never reached the unchanged
`0.25` gate. E21 native motion retained `0.36898` and passed its independent
endpoint audit. This closes the coarse fixed-connectivity ambient-map family;
the next hypothesis must change representation or mechanism, not its profile.

E22's treatment materially improved public truth distance and held-out images,
but the official report contract failed and its nonpromotable endpoint missed
the inherited directional-normal fidelity gate. E23's full-rank coordinates
passed their algebraic claims but only delayed complete-path rejection. E25
therefore tests a genuinely new surface/normal representation: a visual-hull
initialized multiresolution SDF, continuous SDF-gradient normals, and
differentiable FlexiCubes, with nonpromotable topology search and exact
stage-boundary commitment. Its reusable implementation, complete local public
preflight, and corrected build-only CGAL 6.2 image gate passed. Its sole L40S
worker then failed before training on a CUDA-geometry/CPU-camera-intrinsics
device mismatch. The zero-retry rule closes E25; do not propose a device-fix
rerun. Useful retrospective review topics include:

- continuous gradient-normal and inverse-transpose transport correctness;
- matched control/evidence equality and truth-only evaluation;
- topology-search isolation and the post-commit connectivity freeze;
- exact stage-boundary topology, probe, replay, and protected-read guards.

B3 should be audited separately as a process-isolation baseline. It may not
change scientific source or schedule, retry B2, put Python 3.8 on global
`PATH`, substitute a proxy endpoint, or be described as a causal E25 ablation.
Its corrected non-GPU deployment passed the legacy-import, control-import, and
process-isolation checks, and its public input passed exact binding audit. The
single dependency call then failed during remote GPU-container module
hydration before the function body or scientific probe. The output stayed
empty and no training attempt or endpoint exists. The zero-retry rule closes
B3; do not propose a mount-path fix or rerun under B3. B4's single zero-GPU
source prerequisite failed licensing, SMPL-X/data terms, and evaluator mapping
and is closed before execution. B5's one source prerequisite then passed the
pinned REC-MV source and full 200-epoch schedule but failed mandatory license,
data/template, SMPL, preprocessing, and evaluator bindings. It is also closed
before execution. B1-B5 are terminal; do not activate another baseline from
review.

Treat the accepted 0902 roadmap as terminal at its declared stops. No current
experiment is active, and a review may recommend only materially new public-
first hypotheses for a separately accepted roadmap; it cannot reactivate a
closed ID.

After that terminal roadmap, V2 qualified fixed-camera human phase, semantic
and uncertainty evidence, a robust visual hull, local material tracks, and a
shortcut-resistant direct-field implementation. The sole 600-step G02 science
attempt nevertheless degraded held-out silhouette IoU by about `0.063` versus
its matched control and produced an eight-component, Euler-eight raw zero set.
G02 and its strict downstream chain are closed.

A frozen V1 posed replay is the visual baseline: it follows broad pose but is
coarse, thin, and untextured. G03 static appearance stopped on its training
SSIM screen without development scoring. G04 phase-conditioned appearance then
passed target-excluded training checks and one frozen development evaluation,
including `26.43%` RGB-MAE improvement and `0.05761` crop-SSIM improvement over
neutral with unchanged foreground. Audit its leave-one-out exclusion,
target-RGB boundary, phase capacity, and metric aggregation. Its independent
blinded human preference gate remains pending, and a better-looking video is
not evidence that the layered-canonical goal has been achieved.

D01 and D02 later stopped at topology gates. D03 produced an exact-topology
closed implicit body but failed its single development IoU/boundary gate. L03
uses that body only as prior-derived initialization under a new ID. Its public
open-layer/training fixtures, train-only semantic audit, deterministic real
upper/lower initialization, and one local training/checkpoint step pass. Audit
whether the registered boundaries, contact, clearance, outward displacement,
and target-CUDA plan prevent mask-only improvement from being mistaken for a
human-like layered reconstruction. Target CUDA and science remain pending.

Do not suggest retuning a closed lane after seeing its failure, combining
closed treatments without registering their interaction as the hypothesis, or
using the sealed test for selection.

## Recommended reading order

1. `README.md` for scope and headline status.
2. `docs/PROJECT_CONTEXT.md` for the implementation and evidence contract.
3. `docs/HANDOFF.md` for completed work, failures, and stop decisions.
4. `configs/evaluation/post_v1_experiments.yaml` for the structured post-V1 results.
5. `docs/ROADMAP.md` for unaccepted future directions.
6. The source and tests mapped below.

## Source map

| Question | Primary implementation | Contract and public fixture |
|---|---|---|
| Certified cage projection | `src/frayid/feasible_cage.py` | `tests/test_feasible_cage.py`, `scripts/run_post_v1_e4_qp_preflight.py` |
| Safe RMS and camera scaling | `src/frayid/geometry.py`, render/training modules | camera, renderer, geometry, and training tests |
| Continuous-time motion | motion modules under `src/frayid/` | `scripts/run_post_v1_e1_preflight.py` |
| Feasible-direction comparison | `src/frayid/feasible_cage.py` | `scripts/run_post_v1_e2_preflight.py` |
| Exact-distance field support | SDF modules under `src/frayid/` | `scripts/run_post_v1_e3_preflight.py` |
| Material-point evidence | `src/frayid/material_tracks.py` | `tests/test_material_tracks.py` |
| Exact replay state | `src/frayid/replay_state.py` | `tests/test_replay_state.py`, `scripts/run_post_v1_replay_state_preflight.py` |
| Exact collision source audit | `tools/e7_cgal/` | `scripts/run_post_v1_e7_source_lineage_audit.py` |
| Opaque visibility contract | `src/frayid/renderer_contract.py` | `scripts/modal_post_v1_e8_opaque_gate.py` |
| Tracklet outlier process | `src/frayid/material_tracks.py` | `tests/test_tracklet_outlier.py`, `scripts/run_post_v1_e9_preflight.py` |
| Embedded carrier public gate | `src/frayid/embedded_carrier.py`, `tools/e10_cgal/` | `tests/test_embedded_carrier.py`, `scripts/run_post_v1_e10_preflight.py` |
| Renderer determinism trace | `src/frayid/renderer_determinism.py`, `src/frayid/triangle_rasterizer.py` | `tests/test_triangle_rasterizer.py`, `scripts/run_post_v1_p1_macos_diagnostic.py` |
| Interface-conforming field | `src/frayid/interface_field.py`, `tools/e6_cgal/` | `tests/test_interface_field.py`, `scripts/run_post_v1_e6_preflight.py` |
| Topology-safe SDF bridge | SDF and topology modules under `src/frayid/` | `scripts/validate_topology_safe_sdf.py` and related tests |
| Coarse exact ambient maps | `src/frayid/coarse_orientation_map.py`, `src/frayid/composed_orientation_map.py`, `src/frayid/active_tangent_orientation_map.py` | corresponding E19-E21 runners and tests |
| Eulerian image-active zero set | `src/frayid/differentiable_isosurface.py`, `src/frayid/eulerian_field.py`, `src/frayid/eulerian_reconstruction.py` | `tests/test_differentiable_isosurface.py`, `tests/test_g22_public_gate.py`, `scripts/run_post_v1_g22_public_gate.py` |
| Full-rank intrinsic coordinates | `src/frayid/intrinsic_geometry.py` | `tests/test_intrinsic_geometry.py`, `tests/test_e23_public_gate.py`, `scripts/run_post_v1_e23_public_gate.py` |
| E25 public representation | `src/frayid/normal_integrable_sdf.py`, `src/frayid/flexicubes_adapter.py`, `src/frayid/e25_stage.py`, `src/frayid/e25_public_fixtures.py` | related tests and `scripts/run_post_v1_e25_public_preflight.py` |
| E25/B3 registered contracts and status | `configs/evaluation/post_v1_experiments.yaml` | public roadmap and handoff |
| V2 fixed-camera, evidence, field, and posed-baseline code | `src/frayid/v2/` | corresponding `tests/test_v2_*.py` synthetic and contract tests |
| V2 D03/L03 body and open-layer successor | `src/frayid/v2/d03_implicit_body.py`, `src/frayid/v2/l03_open_layers.py`, `src/frayid/v2/l03_training.py`, `src/frayid/v2/l03_modal.py` | `tests/test_v2_d03_implicit_body.py`, `tests/test_v2_l03_open_layers.py`, `tests/test_v2_l03_training.py` |

## Review constraints

- Preserve the fixed V1 metrics, topology gates, and train/development/test
  separation.
- Treat the certified E4 solver as retained numerical infrastructure, not as a
  promoted geometry method.
- Require synthetic and local falsification before private or paid execution.
- Keep outputs immutable, automatic paid retries at zero, and failure reports
  structured.
- Never request private media, body data, meshes, checkpoints, hashes, cloud
  identifiers, or absolute paths.
- Do not interpret an output file, a visually plausible render, watertightness
  alone, or a falling training loss as scientific success.

## A useful review response

A strong review should state one mechanism, explain why existing evidence does
not already reject it, define a matched control, name the public synthetic
fixture, retain the unchanged geometry and topology gates, set a compute and
retry ceiling, and give an unambiguous stop condition. It should also identify
which closed experiments it supersedes or differs from.
