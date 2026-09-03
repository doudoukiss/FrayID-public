# FrayID

FrayID is a research-only prototype for recovering one canonical clothed human
surface from a cooperative monocular self-rotation video.

This public repository contains source code, tests, and a privacy-safe account
of the reconstruction pipeline and its V1 research result. It intentionally
contains no human videos, body data, model weights, checkpoints, meshes,
generated media, private run identifiers, or artifact fingerprints.

## Start here

Read these documents in order when reviewing or continuing the project:

1. [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) explains the problem,
   architecture, evidence contract, and acceptance criteria.
2. [`docs/HANDOFF.md`](docs/HANDOFF.md) records what was implemented, which
   approaches failed, and the accepted V1 result.
3. [`docs/AI_REVIEW_GUIDE.md`](docs/AI_REVIEW_GUIDE.md) gives an AI reviewer a
   concise reading order, source map, and decision boundary.
4. [`configs/evaluation/post_v1_experiments.yaml`](configs/evaluation/post_v1_experiments.yaml)
   provides the E0-E26, B1-B5, and P0-P11 state in machine-readable form.
5. [`configs/evaluation/current_project_track.yaml`](configs/evaluation/current_project_track.yaml)
   records the corrected reproduction-first target and claim boundaries.
6. [`docs/ROADMAP.md`](docs/ROADMAP.md) separates completed work from possible
   post-V1 research.

Together with the source and tests, these files are intended to provide enough
context for a web-based code reviewer or AI assistant to understand the project
without access to private data.

## Current reproduction-first direction

The primary track is now an official/reference SelfRecon-style reproduction:
one unified canonical implicit outer surface, deformed into the observed frames
by skeletal motion plus bounded residual non-rigid motion. This is the same
class of target as the original V1 contract. It does not require skin, shirts,
trousers, and hair to be reconstructed as separate physical objects.

Later work correctly exposed limitations of radial clothing shells, but then
made explicit garment curves, material coordinates, contact, rest metric, and
strain prerequisites for all progress. That stronger MANTLE target is preserved
as deferred optional research. It no longer blocks reproducing and evaluating
the unified outer surface.

New capture uses a staged ladder: bare torso for camera/phase/pose/SDF plumbing,
a close-fitting top for the first clothed outer-surface test, and loose clothing
as a later stress test. A bare-torso result is never garment evidence.

R01 is now registered. Its local official-source and public-example binding
passed. The sole authorized private image build then exhausted hosted-runner
disk before publication, with zero retries, so the required content-addressed
runtime still does not exist. No build, runtime import, GPU worker, training,
project-evidence read, or capture is active. Any successor must be separately
registered with an explicit disk-capacity preflight and owner authorization.

## Pipeline

```text
local self-rotation video
  -> quality-ranked train/held-out frames
  -> Sapiens2 masks and normals
  -> CameraHMR/SMPL initialization
  -> shared camera/body refinement
  -> canonical SDF and topology-safe explicit carrier
  -> skeletal plus residual deformation
  -> held-out geometry evaluation
```

CameraHMR is used only for initialization, Sapiens2 provides image evidence,
SMPL provides the articulated scaffold, and the SDF represents the canonical
clothed surface. Proxy masks, constant normals, proxy cameras, zero poses, and
the existence of an output file are not accepted as reconstruction evidence.

V1 is geometry-only. It does not provide texture, appearance, measurements,
sizing, garment inference, virtual try-on, or commercial readiness.

## V1 status

V1 is closed and its image metrics were accepted against a previously sealed
test split. The evaluated canonical SDF achieved silhouette IoU `0.8700`, an
improvement of `+0.2043` over initialization, normalized boundary error
`0.00483`, and median normal error `20.44°`. A post-V1 audit later found that
the historical extraction's zero-flip claim had used a rebased reference; a
separate original-reference correction passed the unchanged development and
topology gates without reopening the sealed test.

The registered post-V1 hypotheses and correctness probes were subsequently
evaluated under sequential stop rules. E1-E3 stopped at their declared motion,
field-support, and QP gates. E4
replaced the defective experimental QP kernel with a certified mature solver;
its numerical and synthetic gates passed, but both matched real smoke arms
missed the objective and exact-resume requirements. E5's public material-track
oracle improved several geometry measures but missed its normal, robustness,
and resume gates. E6's public interface-conforming field passed, but the
accepted source failed the required exact self-intersection audit. P0 repaired
exact checkpoint replay; E7 then found no exact embedded same-connectivity
source, E8's opaque treatment failed real-smoke replay, and E9 found no robust
lambda that passed both corrupted controls. E10-E15 and P1-P11 then tested
embedded carrier construction, exact genus/fidelity contracts, collision-path
oracles, and deterministic reductions. The final P11 singleton Planar-DAT
filter was fast and bitwise repeatable, yet its combined endpoint contained
317 exact self-intersection pairs and its full-path oracle timed out. This
closed arithmetic, batching, tolerance, sampling, and timeout variants of that
family.

G16/E16 next exceeded its fixed 7,200-second public cap while constructing the
exact fine ambient scaffold. E17's coarse bi-Lipschitz fallback retained only
`0.01626` of tangential motion. The E19-E21 exact determinant sequence raised
retention from `0.07463` to `0.09314` and then `0.13660` with an active-tangent
direction, still below the unchanged `0.25` gate. Native E21 motion retained
`0.36898` and passed its independent exact endpoint audit. This closes the
coarse fixed-connectivity ambient-map family under the registered profiles.
The isolated B1 SelfRecon baseline also closed before training because its
separately licensed assets and runtime prerequisites were not bound. E18
remains blocked. E22's one official public attempt closed on report assembly
and absolute-fidelity gates; its retained endpoint audit was nonpromotable.
E23's full-rank intrinsic coordinates passed matrix, gradient, and reachability
checks but closed when both frozen optimization arms eventually rejected a
complete path before replay or truth comparison. B2 then bound the genuine
male model and official 689-frame public example, but its single dependency
worker failed before the probe body when legacy Python shadowed the required
control-plane runtime; B2 closed without retry or training.

E25 was the final registered native scientific lane. It tested a visual-hull
initialized multiresolution image-active SDF with continuous gradient normals
and differentiable FlexiCubes against the frozen G22 control, using no RGB,
tracks, learned motion, or protected evidence. Its reusable implementation,
complete local public preflight, and corrected non-GPU image build passed. Its
sole public L40S worker then failed before training because CUDA target geometry
met CPU camera intrinsics; E25 closed without retry. B3 was a separate dual-runtime
SelfRecon public-baseline prerequisite and is not a B2 retry; its comparison
must disclose that it uses RGB. B3's corrected non-GPU image passed the
legacy imports, control-plane import, and process-isolation checks, and the
public inputs passed a complete bytewise binding audit. Its single A10G call
then failed during GPU-container module hydration before the dependency body or
scientific probe. No marker, training attempt, epoch, or endpoint exists, and
the zero-retry contract closes B3. E18/E24/E26 remain blocked. D3-Human's one
zero-GPU prerequisite then failed licensing, SMPL-X/data terms, and evaluator
mapping and closed before execution. REC-MV's one source-only prerequisite
passed source/schedule identity but failed license, data/template, SMPL,
preprocessing, and evaluator bindings. B1-B5 are terminal with no external
candidate active.
Proxies, pretrained endpoints,
shortened runs, and sealed-test reuse do not qualify.
These results describe one private research sequence, not a population
benchmark.

The accepted 0902 roadmap is terminal at its declared stop conditions. E25
produced no certified representation, so private E25 and E26 never became
eligible; no registered post-V1 lane is active. Further experiments require a
separately accepted, materially new roadmap.

V2 subsequently tested a layered-canonical route. Fixed-camera human phase,
semantic/uncertainty evidence, a robust visual hull, local material tracks, and
direct-field engineering checks passed. The one 600-step G02 scientific
attempt failed independent silhouette and topology gates, so its downstream
normal/layered/multimodal stages were not activated. A frozen V1 posed replay
now provides an honest visual baseline: it follows broad pose but remains
coarse, too thin, untextured, and unlike a realistic person. The private media
and replay are not part of this repository.

The separately registered G03 static-appearance treatment stopped before
development scoring because its training SSIM improvement missed the frozen
gate. G04 then passed a 144-frame leave-one-out screen and one frozen 36-frame
development evaluation: RGB MAE improved by `26.43%` and crop SSIM by `0.05761`
over the neutral baseline, with identical foreground and no development RGB in
fitting. The 180-frame replay is an automated passing, sequence-specific
presentation result; its blinded human preference gate remains pending. It is
not proof of authoritative layered geometry, and no identity-bearing media are
included here.

A later geometry-only successor sequence stopped D01 on its train topology
precheck and D02 on exact self-intersection. D03 produced a deterministic,
closed, outward, watertight one-component Euler-2 capsule-tree body, but its
single development evaluation failed the absolute IoU and boundary gates. L03
therefore treats that body only as prior-derived initialization evidence. L03
has passed public open-layer fixtures, train-only upper/lower semantic support,
deterministic open-layer initialization with registered boundaries/contact,
and one local training/checkpoint step. Its target-CUDA qualification and every
scientific attempt remain pending; no authoritative layered result exists.

## Local setup

```bash
uv sync --extra dev
uv run frayid doctor
uv run python -m pytest -q
```

To use local media, copy `configs/assets/local_media.example.yaml` to
`configs/assets/local_media.yaml`, fill in metadata calculated from your own
video, and keep both that manifest and the media local. The public project does
not ship a runnable private dataset or pretrained model assets.

Public synthetic experiment runners are available under `scripts/`; exact-
predicate and ambient-scaffold reference tools are under `tools/`. They exercise
public procedural fixtures only and do not reconstruct the private subject.
E25's public preflight runner is included; its operator-only matched GPU runner
and private artifact bindings are intentionally excluded.

## Privacy and licensing

Read [`PRIVACY.md`](PRIVACY.md) before supplying data. Human media and derived
geometry must remain local and must not be committed.

The FrayID source in this repository is licensed under Apache-2.0. The design is
informed by research systems and third-party models with their own licenses and
possible patent considerations. Review all upstream terms independently before
commercial use.
