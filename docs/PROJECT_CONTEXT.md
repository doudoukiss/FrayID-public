# FrayID project context

Status: public technical overview; reproduction-first unified outer-surface track restored
Scope: canonical clothed-surface geometry V1  
Privacy: no source media, body data, artifacts, or private identifiers

## Problem

FrayID reconstructs one canonical clothed human surface from a cooperative
monocular video in which the subject rotates to expose multiple viewpoints.
The desired output is view-consistent geometry in a canonical pose, not an
independent body estimate for each frame.

The original project failed because it was primarily a collection of
frame-by-frame SMPL inference, fixed-topology repair, and heuristic method
switches. Those components could produce plausible individual frames or mesh
files, but they did not solve sequence-level non-rigid inverse rendering. A
valid reconstruction must explain the whole sequence with shared geometry,
shared camera properties, articulated motion, and measured held-out evidence.

## Goal and exclusions

V1 recovers geometry only. Its accepted output is a canonical signed-distance
field (SDF) and a topology-safe explicit extraction used for rendering and
validation.

The current primary target retains that original V1 meaning: a SelfRecon-style
unified outer surface with skeletal and bounded residual non-rigid deformation.
It does not require separated garment meshes, explicit neckline/cuff/hem loops,
material coordinates, contact state, rest metric, or cloth strain. Those may be
regional diagnostics or separately qualified later research; they do not block
the reference reproduction.

V1 intentionally excludes:

- RGB appearance, texture, relighting, and material recovery;
- body measurements, absolute scale, sizing, tailoring, and garment patterns;
- garment classification or semantic clothing inference;
- avatars, virtual try-on, product UI, and commercial deployment;
- population-level generalization claims.

The project is research-only. Upstream models, research implementations, and
possible patents have separate terms that require independent review before
commercial use.

For method development, a bare-torso regional capture may test camera, phase,
pose, SDF, and deformation plumbing, but it cannot support a clothed-surface or
garment claim. A close-fitting top is the first staged clothed case.

## Responsibility boundaries

| Component | Responsibility | Not authoritative for |
|---|---|---|
| CameraHMR | Initial per-frame pose, shape, translation, and camera estimate | Final geometry or final evidence |
| Sapiens2 | Observed masks, surface normals, and body joints | Canonical shape or camera parameters |
| SMPL | Shared articulated scaffold, joints, LBS transforms, and skinning frame | Clothed surface detail |
| Shared initialization fit | Shared body shape and camera intrinsics; framewise pose/root/translation refinement | Final clothing geometry |
| Canonical SDF | Final implicit clothed-surface representation | Texture, scale, or garment semantics |
| Explicit carrier | Differentiable rendering, deformation support, and topology-safe SDF extraction | Independent proof of success |
| Residual deformation | Bounded frame-conditioned non-rigid motion not explained by LBS | Unrestricted per-frame shape replacement |
| Evaluation | Held-out silhouette, boundary, normal, generalization, and topology gates | Training or candidate tuning from sealed test data |

## Evidence contract

The active pipeline requires real Sapiens2 masks and normals plus real
CameraHMR/SMPL initialization. It rejects proxy masks, constant normals, proxy
cameras, zero poses, and initialization meshes presented as trained output.

The private development sequence was sampled into 180 distributed frames: 144
for training and 36 interleaved held-out frames. Training is blocked when fewer
than 120 usable frames pass quality and evidence validation. A separate
36-frame source split was sealed before its evidence or metrics were opened and
was used once for the final test.

Only the roles and counts are public. The source media, exact media metadata,
hashes, frames, body evidence, and derived geometry remain private.

## Data flow

```text
local rotation video
  -> quality-ranked distributed frame selection
  -> immutable train / held-out split
  -> Sapiens2 masks, normals, and observed joints
  -> CameraHMR / SMPL initialization
  -> shared shape and camera plus framewise pose refinement
  -> canonical explicit carrier and SDF
  -> SMPL/LBS plus bounded residual deformation
  -> silhouette, boundary, and normal inverse-rendering losses
  -> topology-safe SDF extraction
  -> held-out validation
  -> single sealed-test evaluation
```

## Geometry and optimization

The implementation contains:

- a positional-encoding neural SDF and Eikonal regularization;
- periodic Marching Cubes extraction with dynamic bounds;
- an explicit canonical carrier coupled to the SDF;
- SMPL/LBS skeletal deformation;
- frame-conditioned residual deformation and bounded root corrections;
- silhouette, boundary, normal, SDF/mesh consistency, deformation Jacobian,
  temporal, orientation, area, edge-strain, and smoothness losses;
- coarse, medium, and fine schedules;
- checkpoint/resume and immutable run provenance;
- scale-aware rendering and visibility-aware normal comparison;
- topology-preserving optimizer projection with backtracking;
- topology-safe unsigned closest-distance plus occupancy-sign narrow-band SDF;
- deterministic extraction and Chamfer, normal, volume, watertightness,
  connected-component, face-orientation, and face-collapse tests.

## Why topology preservation matters

A watertight mesh can still be invalid when faces fold inward or collapse.
Earlier work appeared watertight but contained thousands of orientation flips,
which made signed-distance conversion fragment or produce wrong normals.

The current optimizer therefore evaluates every proposed canonical update
against the original face orientation and minimum face area. Unsafe updates are
backtracked; optimizer moments are damped when projection shortens a step.
Checkpoint writing, reconstruction evaluation, and SDF promotion repeat the
topology checks independently.

For SDF conversion, sign comes from supersampled conservative occupancy while
distance near the surface comes from unsigned closest-triangle distance. Far
from the surface, a distance transform is sufficient. The final explicit
surface preserves the accepted carrier connectivity while projecting vertices
toward the SDF zero level under the same topology gate.

## Coordinate and rendering contract

The tests cover camera coordinates, OpenCV projection, crop transformations,
root rotations, and virtual-camera equivalence. Surface normals are converted
from the renderer/OpenCV convention to the Sapiens2 observation convention
before comparison. Rendering uses depth-visible surface samples and
reference-density normalization so metrics converge as sample count increases.

The renderer is tested for finite, useful silhouette and normal gradients. A
renderer implementation is not accepted merely because a picture looks
reasonable; its metric stability and optimization direction must be verified.

## Gates

Initialization requires median silhouette IoU at least `0.75`, shared shape and
focal length, and no root flips or framewise focal drift.

The fixed geometry acceptance gates are:

| Metric | Requirement |
|---|---:|
| held-out silhouette IoU | at least `0.85` |
| improvement over initialization | at least `+0.10` |
| normalized boundary error | at most `0.015` |
| median normal error | at most `25°` |
| train/held-out IoU gap | at most `0.05` |
| dominant component area fraction | at least `0.98` |
| extracted geometry | deterministic, watertight, one component, no flips/collapses |

The cloud smoke gate uses 24 frames and two epochs with no automatic paid
retry. Every enabled loss must be finite and active, all expected parameter
groups must change, resume must work, topology must remain valid, and the fixed
objective must fall by at least 5%.

## Accepted V1 outcome

| Representation and split | IoU | Init. IoU | Improvement | Boundary | Normal | Topology |
|---|---:|---:|---:|---:|---:|---|
| explicit carrier, development validation | `0.8777` | `0.6489` | `+0.2289` | `0.00454` | `21.38°` | pass |
| canonical SDF, development validation | `0.8776` | `0.6489` | `+0.2288` | `0.00456` | `21.37°` | pass |
| canonical SDF, sealed test | `0.8700` | `0.6658` | `+0.2043` | `0.00483` | `20.44°` | pass |

The final train/test IoU gap was `0.0152`. The historical report described one
watertight Euler-2 component with zero detected face flips and zero collapsed
faces. A later original-reference audit found one flipped face that the
historical rebased-reference check had hidden. A separately versioned
original-reference correction subsequently passed zero-flip, zero-collapse,
watertightness, and unchanged development gates. The sealed test was not
reopened and its metrics remain historical rather than evidence for selecting
the correction.

These numbers describe one private cooperative sequence. They demonstrate that
the implementation met its declared V1 contract; they do not establish broad
generalization.

## Post-V1 result

An incumbent-binding audit and a serial set of independent registered hypotheses were
evaluated after V1. The audit exposed and corrected the original-reference
topology issue described above. E1-E3 then stopped at their predeclared gates:

- a continuous-time cubic motion representation was slightly worse than its
  equal-budget slot control on omitted synthetic times;
- expanding the exact-distance support band improved coverage but missed its
  fixed residual limit and retained an invalid raw zero set;
- a linearized feasible cage direction passed a strong synthetic fixture, but
  its custom active-set QP failed to converge in the single real-data smoke.

Public counterexamples subsequently reproduced a deterministic inequality-QP
defect. E4 replaced that kernel with a mature solver and independent
feasibility/optimality certificates. More than one thousand public numerical
cases passed, as did the synthetic CUDA and corrected-incumbent gates. The
matched real treatment completed every certified projection, but its objective
improvement was `4.979%` against a fixed `5%` gate, while the matched control
improved `3.050%`; both arms also failed exact next-step resume. E4 therefore
closed without a bounded comparison. The certified solver and common numerical
corrections remain correctness improvements, not a promoted geometry method.

E5 tested fixed material-point tracks as train-only cross-frame evidence. On a
public oracle sequence, tracks improved canonical Chamfer by `27.2%`, posed
motion error by `24.4%`, silhouette, and boundary. It nevertheless missed the
fixed `2°` pooled-normal improvement gate, failed exact resume in every arm,
and became worse than control under the registered corrupted-track fixture.
The lane closed before any private tracking, real training, or development
evaluation.

E6 tested an exact-predicate interface-conforming piecewise-linear field. All
ten valid public fixtures and six negative fixtures passed, including
near-contact, sign, zero-subcomplex, deterministic extraction, watertightness,
and Euler checks. The unchanged accepted explicit carrier then passed ordinary
manifold, winding, watertight, one-component, and Euler-2 checks but failed an
exact global self-intersection audit. The registered contract forbade repair,
so E6 closed before private field construction or development evaluation.

P0 then repaired the common state-transition contract. Version-2 checkpoints
capture model and optimizer ordering, batch permutation/cursor, Python, NumPy,
CPU/CUDA and named-generator RNG, scheduler/scaler, stage, and auxiliary state.
CPU and one L40S produced bitwise-identical next updates at every registered
resume boundary; a deliberately mismatched RNG/cursor control diverged.

E7 required an exact embedded same-connectivity source before collision-aware
optimization. Exact-predicate audits found intersections in every eligible
lineage candidate, so E7 stopped without repair, IPC, training, or evaluation.

E8's opaque visibility path passed hidden-surface, triangle-order, crop/camera,
normal-axis, thin-gap, finite-difference, and known-shape public gates. In the
paired real smoke, the control replayed exactly while the opaque treatment did
not, so the lane closed without bounded or development evaluation.

E9 evaluated the eliminated tracklet penalty `lambda*S/(lambda+S)`. Clean
tracks improved canonical Chamfer `72.57%`, motion RMSE `47.73%`, and pooled
normal error by `17.35°`, with exact replay. No fixed lambda was simultaneously
no worse than no-track and mean-pseudo-Huber controls on corrupted evidence,
so no tracker weights or private RGB were used.

E10 then tested a new-connectivity embedded carrier. A public CGAL Alpha
Wrapping grid froze one pitch-normalized parameter pair using only synthetic
fixtures. The one real-source construction was byte deterministic, exactly
non-self-intersecting, watertight, outward oriented, one component, and fully
contained the registered source probes. It nevertheless had Euler number
`-4`, not the required `2`. E10 closed before transfer, pose checks, or
development evaluation; no alternate parameter or repair was attempted.

P1 added renderer-stage traces and a first-bitwise-difference reporter. The
macOS off-centre CPU fixture repeated bitwise 100 times. A later single-L40S
gate found every registered forward stage bitwise stable across 100 repeats,
while the geometry gradient first differed on the second backward pass by about
`7.45e-9`; the cage gradient inherited that difference. Checkpoint-v2 next-step
replay passed at all four registered checkpoints. P1 closed at its exact
bitwise gate, the renderer operator remained unchanged, and E8 was not reopened.

E11 then tested only whether a genus-zero constructor could produce exact valid
topology before private geometry. Its five public procedural fixture classes
all produced byte-identical repeated serializations and passed an independent
exact audit for zero self-intersections, one watertight outward component, and
Euler number 2. This is not a fidelity result: concavity, gap, normal, distance,
and volume preservation remain unproven, and no private source was read.

A frozen follow-up public fidelity gate resolved that uncertainty. Convex,
rigid, and scale controls passed, but a concave pocket exceeded the volume and
feature-stratum limits. A one-component Euler-2 hairpin lost every registered
external gap and strongly distorted volume and surface normals. The constructor
was therefore rejected as a faithful carrier source despite retaining exact
Euler-2 topology and determinism. No cloud replication or private carrier read
followed the local failure.

Later public-only collision research ended at P4. P3 retained 107,581 dense
hairpin near pairs and exceeded its narrow-phase time gate. P4 then tested IPC
Toolkit 1.6.0 Planar Divide-and-Truncate, but stopped before mesh construction
because the fixed absolute-tolerance Tight-Inclusion oracle rejected 82 of
1,024 analytic paths. Every rejection was a small-scale tangent case;
crossing, separating, and zero-motion cases had none. This is an oracle
disagreement, not an exact intersection proof, so P4 makes no hairpin safety or
complexity claim and no later scientific lane is active.

P5 calibrated a unit-frame Tight-Inclusion judge against the same 1,024 known
analytic paths. It matched every raw collision/safety label, accepted every
Planar-DAT-filtered path, repeated deterministically, and reproduced the 82 old
absolute-tolerance rejections as a diagnostic. P5 is an evaluator pass only;
mesh-level Planar-DAT safety and complexity remain untested.

P6 performed that mesh test on the exact P2 hairpin. Both frozen Planar-DAT
mechanism calls exceeded 30 seconds before normalized TI or endpoint audit, so
P6 closed without a mesh-path result. The remaining bottleneck is now before
the oracle, inside dense trust-region construction/filtering.

P7 tested the cheap isotropic trust filter. Its nested supervisor included
startup overhead, but an isolated profile showed the decisive result: all
5,298 wrap vertices were restricted and only 2.36% of tangential motion
remained, below the frozen 25% floor. P7 closed without retry.

P8-P11 then removed Planar-DAT's implementation ambiguities in stages. P8
matched 24 mixed-stencil cases exactly but could not obtain a full upstream
hairpin reference within five minutes. P9 found seven scalar/vector branch
disagreements. P10 showed that even 183 apparent consensus candidates differed
from isolated upstream. P11 therefore queried all 194,312 candidates through
singleton upstream filters. Two full passes took about four seconds each and
repeated bitwise, but the combined endpoint contained 317 exact
self-intersection pairs and the independent full-path judge exceeded 120
seconds. The complete Planar-DAT execution family is closed: local candidate
decisions do not compose into a safe global refined-mesh update.

G16/E16 then attempted the ranked exact ambient-scaffold route. The retained
fine constrained construction could not build the complete closed-box
tetrahedral complex inside the fixed 7,200-second public cap, so E16 closed
before image, private, development, GPU, cloud, or paid work. E17's separately
registered coarse bi-Lipschitz fallback passed its safety machinery but kept
only `0.01626` of the tangential proposal against the unchanged `0.25` floor.

E19-E21 tested whether exact determinant maps could recover that motion without
weakening the complete-path certificate. One coarse block retained `0.07463`;
four repeated residual blocks retained `0.09314`; four active-determinant
tangent blocks retained `0.13660`. E21's native proposal retained `0.36898` and
passed an independent exact endpoint/nesting audit, but the tangential proposal
remained below its frozen gate. The coarse fixed-connectivity ambient-map
family is therefore closed under the registered profiles. E18 remains blocked.

The isolated B1 SelfRecon prerequisite verified the pinned public source and
configuration but could not bind the separately authorized assets, data, and
Linux/CUDA runtime required for a valid full baseline. It closed before
training rather than substituting proxies.

No post-V1 treatment was promoted, tuned after its stop, or evaluated on the
sealed test. E22/G22 put a fixed-domain Eulerian piecewise-linear zero set in
the public image-loss graph. Its one official attempt closed after report
assembly failed; a read-only retained-endpoint audit also found that the
treatment missed the inherited directional-normal fidelity gate despite a
large truth-distance and held-out-IoU improvement. It was not rerun.

E23 then changed only the optimization coordinates to the invertible full-rank
map `U=(M+L)V`. Its matrix, image-gradient, and reachability checks passed, but
the frozen direct and intrinsic arms both eventually exhausted complete-path
backtracking before the replay point. E23 therefore closed before its 300-step
truth comparison. At that terminal checkpoint no native scientific lane was
registered. E24
shared RGB evidence remains blocked because no passing executable
representation was frozen.

B2 is separate operational baseline work. It may run only after the exact
licensed assets, official public sequence, and pinned legacy runtime are bound,
and it must complete the upstream 200-epoch path. The genuine male model and
official 689-frame RGB/mask/normal example are now present in ignored local
storage without an extracted pretrained endpoint. The compatible Linux/CUDA
runtime and compiled extensions remain the blocking prerequisite. A first
image build stopped at an interactive operating-system timezone prompt before
any GPU worker began. A second build passed that correction but stopped before
GPU execution when historical OpenMesh and pycocotools builds required CMake
and a compatible Cython API. The current build-only correction supplies those
prerequisites without changing SelfRecon source, runtime versions, inputs, or
its 200-epoch command. That image completed, but the app was rejected before
worker dispatch because its conditional training disk request was below the
current platform minimum. The next deployment changes only that resource
request; at that checkpoint no GPU attempt had started.
The corrected deployment reached one L40S worker, but Modal bootstrap selected
the legacy Python 3.8 environment for its own runner and exited because the
current runtime requires Python 3.10 or newer. The probe body and attempt marker
did not run. A GPU worker nevertheless started, so the frozen zero-retry rule
closes B2 before its 200-epoch training and before any private adapter.

E25 was the final active native scientific lane. It tested whether E22's
strong image-to-zero-set signal can meet the missing normal-fidelity gate when
the surface is represented by a visual-hull initialized multiresolution SDF,
rendered with continuous SDF-gradient normals, and extracted with
differentiable FlexiCubes. The first comparison remains mask/boundary/normal
only; RGB, tracks, learned motion, private evidence, development feedback, and
the sealed test are unavailable. Topology-search intermediates are
nonpromotable, exact audits bind stage boundaries, and committed connectivity
freezes before certified refinement.

The reusable visual-hull, neural-SDF, NeuS sampling, continuous-normal,
integrability, FlexiCubes, stage-commitment, replay, fixture, and read-guard
implementation is complete. The one local public preflight passed all 15
fixture-stage audits, five final commitments, the 10,592-face P2 regression,
derivative/transport controls, replay, and six protected-read guards. A first
non-GPU image build exposed only an operating-system CGAL version mismatch;
the build-only correction pinned the official CGAL 6.2 release without
changing the experiment, and the corrected image build/import/exact-tool gate
passed. The sole E25 L40S worker then claimed its attempt and failed before
training because CUDA target geometry was projected with CPU camera intrinsics.
No optimizer step or endpoint exists. E25 is closed without retry; its private
stage and E26 are ineligible.

B3 was a new independent SelfRecon process-isolation baseline, not a B2 retry.
It preserves the pinned source, genuine public example, licensed model, and
official 200-epoch schedule while keeping the Python 3.11 control plane
separate from the absolute Python 3.8 child. B3 uses RGB and must be reported
as an external evidence-rich baseline rather than a causal E25 ablation. Its
corrected non-GPU deployment passed legacy-import, control-import, and
process-isolation checks, and its verified public inputs matched the existing
B2 binding byte-for-byte. Its single dependency call then failed during remote
GPU-container module hydration before the function body or scientific probe.
No marker, training attempt, epoch, or endpoint exists. The zero-retry contract
closes B3. E26 remains blocked. D3-Human's single zero-GPU prerequisite audit
then passed its source and official example checks but failed code/third-party
licensing, SMPL-X binding, public-data terms, and evaluator mapping. It closed
before any data/model/GPU work. At this checkpoint REC-MV was eligible but
unregistered; its later terminal source audit is recorded below. New
appearance/track/motion evidence remains deferred.

## Repository map

- `src/frayid/`: reusable geometry, camera, evidence, training, and evaluation
  implementation;
- `tests/`: synthetic and contract tests for coordinates, SDFs, LBS,
  deformation, rendering, optimization, and topology;
- `configs/reconstruction/`: privacy-safe example of the fixed V1 contract;
- `scripts/validate_topology_safe_sdf.py`: local topology-safe SDF validation;
- `src/frayid/differentiable_isosurface.py` and `src/frayid/eulerian_field.py`:
  E22's differentiable zero-set and complete-path certificate core;
- `src/frayid/eulerian_reconstruction.py` and
  `scripts/run_post_v1_g22_public_gate.py`: the frozen matched 12/6-view,
  300-update public comparison and its independent judges;
- `src/frayid/intrinsic_geometry.py` and
  `scripts/run_post_v1_e23_public_gate.py`: E23's full-rank coordinate map and
  terminal public preflight;
- `src/frayid/normal_integrable_sdf.py`, `src/frayid/flexicubes_adapter.py`,
  `src/frayid/e25_stage.py`, `src/frayid/e25_public_fixtures.py`, and
  `scripts/run_post_v1_e25_public_preflight.py`: E25's publishable
  representation and complete public preflight core; the artifact-bound GPU
  control plane remains private;
- `configs/evaluation/post_v1_experiments.yaml`: the sanitized E25/B3
  registration and all preserved terminal decisions;
- `docs/HANDOFF.md`: completed work and rejected approaches;
- `docs/ROADMAP.md`: candidate work after the closed V1 result.

The public repository is intentionally not a reproducibility package for the
private subject. It is a source and technical-context publication.

B5 REC-MV is registered only for one source-only, zero-download, zero-GPU
prerequisite audit of its official 200-epoch public example, licenses, required
assets, and evaluator mapping. This does not authorize reconstruction or a
private adapter.

That audit passed the pinned source and official 200-epoch schedule but failed
the required license, data, template, SMPL, preprocessing, and evaluator
bindings. B5 closed with zero downloads, builds, GPU use, or training; B1-B5
are now terminal.

The 0902 roadmap is complete at its declared stops. E25 produced no certified
representation, private E25/E26 are ineligible, E18/E24/E26 remain blocked,
and no post-V1 experiment is active.

## V2 layered-canonical checkpoint

V2 changed the intended authority from one fused surface to a closed body SDF
plus ordered, evidence-supported open exterior layers. The physical camera was
verified as fixed; the subject performs a non-rigid self-rotation. A fixed-
camera human-phase solution, confidence-aware visual hull, semantic evidence,
and 249 bounded local material tracks passed their registered engineering
checks. The tracks do not form a reliable global clothing identity cycle.

The next direct multiresolution field passed public, CPU, and target-GPU
engineering qualification, including shortcut controls and exact replay. Its
only scientific attempt completed 600 steps, then failed independent
evaluation. Against its matched frozen control, held-out silhouette IoU fell
from `0.63248` to `0.56944`, while boundary error improved from `0.02060` to
`0.01817` and median normal error improved from `30.13°` to `27.51°`. The raw
zero set also had eight components and Euler number eight instead of the
required one component and Euler number two. The strict chain therefore stops;
normal, layered-garment, and multimodal successors were not activated.

A zero-optimizer renderer now replays the frozen V1 geometry through all 180
observed poses as a visual baseline. Its median hard-raster silhouette IoU is
`0.57871` on training and `0.58265` on held-out frames. The broad pose and turn
are visible, but the result is coarse, too thin, untextured, and does not
faithfully recover clothing volume, hair, shoes, or identity appearance. No
media is included in this public repository.

For the weaker source-like and human-like video objective, G03 static canonical
appearance stopped on its training SSIM gate without reading development RGB.
G04 separately registered a smooth train-only turntable-phase appearance with
frozen geometry, pose, camera, rasterization, and foreground. It passed
144-frame leave-one-record-out training checks and one frozen 36-frame
development evaluation: RGB MAE improved `26.43%` and crop SSIM improved
`0.05761` over neutral, while also beating G03 static appearance. No
development RGB entered fitting. Its independent blinded human preference gate
remains pending. Even a human pass would not complete or rename the failed
layered-canonical objective.

D01 and D02 later stopped on train topology and exact self-intersection gates.
D03's replacement capsule-tree body is closed, outward, watertight,
one-component Euler-2, and exactly non-self-intersecting, but its one frozen
development evaluation reached only `0.67428` IoU and `0.015299` boundary error.
It is terminal as a clothed result. L03 reuses it only as prior-derived inner-
body initialization under a new contract. Public layer fixtures, train-only
semantic coverage, deterministic upper/lower open-layer extraction, registered
boundaries/contact, zero exact intersections, and one local gradient/update/
checkpoint step pass. Target-CUDA qualification and science did not run and
are now frozen historical work under the reproduction-first correction.

The current R01 reference-reproduction contract is registered. Its zero-GPU
local audit passed the pinned official SelfRecon source and terms, official
public-example archives, licensed SMPL binding, and exact 689-frame
RGB/mask/normal alignment. R01 remains `built`, not qualified: the required new
content-addressed prebuilt runtime artifact and manifest do not yet exist. No
runtime import, device qualification, training, project-evidence read, or
capture is active.
