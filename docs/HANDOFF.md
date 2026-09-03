# FrayID public handoff

Status: factual, privacy-safe implementation and experiment record  
V1 state: closed and accepted  
Post-V1 state: R01 source/data bound; sole private build failed before artifact publication; no active execution
Private artifacts: intentionally unavailable

> **Current decision (2026-09-03):** the project first reproduces the
> SelfRecon-style unified outer-surface reference path. Earlier chronological
> statements that call L03 CUDA, Q04, MANTLE, or recapture the next eligible
> action are historical and superseded. No capture, CUDA run, paid job, or
> scientific attempt is active. R01 is registered but remains blocked before
> runtime import: its one authorized private build exhausted hosted-runner disk
> before publication and made zero retries. A new build route needs a new
> disk-preflighted registration and explicit authorization.

This handoff explains what was done and what should not be repeated. It omits
media names, media and artifact hashes, cloud application IDs, private volume
names, absolute paths, and subject-derived files.

## Current outcome

The focused repository, data/evidence validation, shared initialization,
geometry training, checkpoint resume, topology-safe SDF conversion, held-out
evaluation, and sealed-test procedure all functioned in the private research
run.

The accepted canonical SDF achieved:

- sealed-test silhouette IoU `0.8700`;
- initialization IoU `0.6658` and improvement `+0.2043`;
- normalized boundary error `0.00483`;
- median normal error `20.44°`;
- train/test IoU gap `0.0152`;
- one watertight Euler-2 component with zero detected flips or collapses.

The test split was opened once after the candidate and thresholds were frozen.
No candidate parameter was updated from the result.

A later audit found that the historical topology report compared the final SDF
carrier with an already-deformed reference. Against the original reference,
one flipped face remained. A separately versioned correction introduced a
small cross-platform safety margin, passed independent CPU and GPU topology
checks, and preserved the unchanged development metrics. The sealed test was
not reopened, and the historical accepted artifact was not overwritten.

## Repository refocus

The project was reduced from a broad product/method collection to one research
objective: sequence-level canonical clothed-surface geometry. Measurement,
sizing, garment, tailoring, avatar, virtual-try-on, Gaussian-splatting,
appearance, and numerous unrelated reconstruction integrations were removed
from the active tree. Their existence in private history does not make them
part of V1.

The supported CLI was narrowed to environment diagnosis, local asset checking,
dataset preparation/validation, initialization fitting/evaluation, and
reconstruction smoke/planning/evaluation. Old schemas and commands received no
compatibility layer.

## Data and initialization

The development sequence used 180 distributed frames with a 144/36 interleaved
train/held-out split. All selected frames had real masks, normals, observed
joints, CameraHMR estimates, and required SMPL deformation inputs.

Initialization jointly fit shared body shape and full-perspective camera
intrinsics while refining framewise pose, root orientation, and translation.
It passed the `0.75` median silhouette-IoU gate at `0.7590`, with boundary error
`0.00945`, shared shape/focal length, and no root or focal-drift blocker.

## Native implementation completed

The codebase implements the canonical SDF, explicit carrier, Eikonal and
geometry losses, LBS, residual deformation, temporal constraints,
coarse/medium/fine training, exact checkpoint/resume, evaluation, provenance,
topology-safe optimization, and topology-safe SDF distillation described in
`PROJECT_CONTEXT.md`.

The official cloud smoke reduced its fixed objective by `10.7%`; every expected
loss and parameter group was active, resume passed, and retries were disabled.

## Experiment progression

### 1. Baseline sequence training

The first full run completed the schedule but stayed near initialization:
held-out IoU was about `0.759`, normal error about `46.5°`, and the SDF was
fragmented and non-watertight. More epochs alone were not the solution.

A second run improved IoU to about `0.782` and normal error to `25.8°`, but the
SDF still had many components and clipped its fixed extraction bounds. The
fine-stage transition caused immediate regression.

Decision: correct rendering/evaluation and extraction contracts before tuning
capacity or duration.

### 2. Renderer and evaluation correction

Three structural errors were corrected:

- compare depth-visible front surfaces instead of averaging front and back;
- convert renderer normals into the Sapiens2 coordinate convention;
- scale surface sampling and splat width with render resolution.

Dynamic SDF bounds and direct evaluation of the extracted SDF mesh were added.
The explicit mesh then reached IoU around `0.828`, while the learned SDF became
a large fragmented shell. This localized the failure to SDF learning rather
than the corrected image evidence.

### 3. Explicit refinement and transform repair

Exact signed-distance distillation removed the giant shell but lost silhouette
and normal fidelity. Explicit refinement reached IoU around `0.835` and then
plateaued; extending the same objective did not help.

Periodic held-out failures were traced to sequence transforms. Replacing every
held-out transform by interpolation degraded otherwise good frames. A
mask-independent median-plus-MAD rule that replaced only temporal SE(3)
outliers reached about `0.843`, still below the fixed IoU gates.

Decision: keep selective transform repair as a diagnostic; do not use blanket
interpolation or select repairs by held-out mask metrics.

### 4. Failed SDF bridges

Several explicit-to-SDF routes were evaluated:

| Route | Observation | Decision |
|---|---|---|
| larger/deeper neural SDF | more components and worse normal fidelity | reject capacity-only scaling |
| dense voxel plus distance transform | watertight but normal error remained high | retain as implementation evidence only |
| higher grid resolution | did not resolve the structural error | reject resolution-only sweep |
| Gaussian smoothing | regressed geometry and could introduce flips | reject |
| direct narrow-band signed distance | unstable sign fragmented the zero set | reject |

The key discovery was that the apparent watertight source mesh contained
thousands of inward-oriented and collapsed faces. Watertightness alone had
hidden a folded canonical surface.

### 5. Topology-preserving canonical optimization

Differentiable orientation, minimum-area, edge-strain, and local-smoothness
losses were added. Every optimizer update was projected by backtracking from
the previous valid state. Checkpoint and evaluation paths independently
rechecked topology.

This produced a canonical explicit checkpoint with zero detected face flips,
zero collapsed faces, and exact checkpoint-to-mesh topology agreement. It
satisfied the explicit topology requirement, although its image metrics still
needed improvement.

### 6. Topology-safe SDF construction

The final local SDF method combined supersampled conservative occupancy sign
with unsigned closest-triangle distance in a narrow band. Synthetic spheres and
rotated clothed ellipsoids passed bidirectional Chamfer, normal, volume,
determinism, component, and watertightness tests.

Direct Marching Cubes of the private high-resolution field still created small
components. The accepted extraction therefore preserved the valid carrier
connectivity and projected it toward the field zero level using trilinear SDF
gradients under the same flip/collapse gate. Source/SDF image metrics then
agreed within the predeclared bridge tolerance.

Decision: keep occupancy sign plus closest distance; do not return to direct
mesh-sign queries or smoothing as a rescue.

### 7. Pose and representation ablations

Bounded root translation modestly improved geometry, while adding root
rotation did not. A robust full-body pose refit greatly reduced joint error but
overfit pose evidence and regressed geometry when run too long. The short fit
was retained as initialization evidence only.

Subdivided free-vertex carriers, alternating optimization, articulated pose
correction, direct dense residuals, exact triangle rasterization, and a
tetrahedral surface were tested. They exposed useful implementation facts but
did not pass their fixed promotion gates:

- free-vertex carriers repeatedly saturated the topology area floor;
- longer alternating optimization improved slowly, then regressed;
- stronger normal weighting traded silhouette for normals;
- articulated correction was unstable at high learning rate and insufficient
  at the stable learning rate;
- zero-initialized dense residuals had low normal error but poor silhouette;
- an exact triangle rasterizer passed synthetic gradient tests but did not
  preserve the established real-sequence metric contract;
- the tetrahedral synthetic gate passed, but real updates were rejected by the
  minimum-area constraint.

Decision: stop each lane at its declared gate instead of retuning from failure.

### 8. Smooth deformation cage

A smooth trilinear deformation cage replaced independent per-vertex motion.
The smoke objective fell by more than `38%` with all updates accepted and valid
topology. A bounded sequence run reduced its objective substantially but still
missed final IoU and normal gates. The cage nevertheless supplied the stable
topological carrier needed for the final motion experiment.

### 9. Normal-only residual refinement

A diffused scalar normal-offset field passed strong synthetic tests, but almost
every real update required topology backtracking and the smoke objective fell
less than 1%. It was not promoted.

Decision: normal-only geometry motion could not resolve the coupled
silhouette/pose error.

### 10. Sequence motion co-adaptation

The successful final step froze canonical geometry and jointly optimized the
frame-conditioned deformer and bounded root corrections using exact sequence
rendering. Its smoke objective fell `12.6%`. The promoted bounded run reduced
the fixed objective `26.3%`, achieved development held-out IoU around `0.876`,
and preserved the zero-flip, zero-collapse watertight carrier.

The accepted explicit surface was converted through the frozen topology-safe
SDF bridge. Development validation measured explicit/SDF IoU
`0.8777/0.8776` and normal error `21.38°/21.37°`, showing negligible bridge
loss.

### 11. Sealed test

A separate 36-frame split with zero development overlap was fixed before
evidence generation. Real CameraHMR and Sapiens2 evidence was generated under
the frozen calibration. The accepted SDF was evaluated once, passed every
gate, and was not modified afterward.

## What not to repeat

- Do not return to per-frame HMR as final geometry.
- Do not lower acceptance thresholds after seeing a result.
- Do not treat a mesh, checkpoint, image, or video as proof of success.
- Do not treat watertightness as sufficient topology validation.
- Do not increase SDF capacity, grid resolution, or training duration without
  a new mechanism and a local fidelity test.
- Do not use direct mesh signed-distance sign in the surface narrow band.
- Do not use Gaussian smoothing as a topology repair.
- Do not interpolate every held-out transform.
- Do not use held-out masks to select individual transform repairs.
- Do not reopen or tune against the sealed V1 test.
- Do not infer texture, physical scale, measurements, garment semantics, or
  commercial readiness from the geometry result.

## Post-V1 registered experiments

### E0 — incumbent binding and original-reference correction

The binding audit reproduced stored metrics but found the hidden
original-reference flip. The first correction was numerically too close to the
signed-area floor to reproduce across platforms. A second correction changed
only the internal projection safety margin and passed local, independent CPU,
and development-only GPU checks. It is a separately versioned corrected
carrier, not a rewrite of the historical sealed result.

### E3 — carrier-covering exact-distance support

The synthetic coverage fixture passed. On the real source, expanded support
covered all requested neighborhoods and substantially reduced maximum field
residual, but still missed the fixed residual threshold and left a
multi-component raw zero set. The lane stopped locally without cloud image
evaluation.

### E1 — continuous-time motion

A fixed cubic-control trajectory and an equal-budget slot baseline were tested
on a public procedural articulated ellipsoid. The cubic treatment regressed
omitted-time vertex RMSE by about `2.56%`; gradient, rigid-zero, frozen-state,
and exact-resume checks passed. The lane closed without tuning knots, control
count, seed, learning rate, or duration after observing the result.

### E2 — feasible cage update direction

A matched real-data diagnostic found that only `66.67%` of cage steps were
accepted at full scale, confirming meaningful backtracking pressure. A
same-cage synthetic QP treatment improved pooled held-out normal error by about
`2.60°` while also improving silhouette, boundary, and signed-gap metrics and
passing exact nonlinear topology validation.

The single real-data treatment smoke then failed because the active-set
halfspace QP did not converge. It produced no checkpoint or structured result,
was not retried, and was not followed by development or sealed evaluation.
Changing the solver or adding a fallback is a new experiment, not an E2 repair.

Public-only follow-up reproduced the solver defect on a one-dimensional
feasible redundant-constraint QP, both row orders of a two-dimensional
extension, and a normalized five-halfspace fixture. Independent optimization
returned feasible optima. The current active-set method can retain incompatible
equality boundaries and exhaust its iteration limit. Float64, row
normalization, or a larger iteration count alone are not sufficient repairs.

This evidence does not include the private smoke constraints and therefore
does not prove the identical failure mechanism occurred there. The accepted
interpretation is that E2's real treatment remained scientifically untested
because its numerical solver failed. E2 stayed closed; its defect motivated
the separately registered E4 experiment below.

### E4 — certified convex projection

A pinned mature inequality-QP solver and independent primal, dual,
stationarity, and complementarity certificates passed 1,000 fixed-seed random
problems plus four fixed regressions. The same-cage synthetic CUDA gate passed
with a `2.90°` pooled-normal gain over the stronger baseline, valid topology,
finite ordinary and zero-residual gradients, and exact resume. A zero-training
bridge also showed that the shared RMS and explicit-camera-dimension fixes
preserved every incumbent geometry gate.

The matched real smoke then closed E4. Control/treatment objective reductions
were `3.050%/4.979%`, both below the unchanged `5%` gate, and both failed exact
next-step replay. Treatment still completed all 48 certified QPs with no
fallback, rejected step, flip, or collapse. No bounded comparison or SDF bridge
followed. The certified solver remains a numerical correctness improvement;
E4 establishes no promoted geometry benefit.

### E5 — train-only material tracks

The public oracle fixture retained 95 material tracks and 3,397 observations.
Treatment improved canonical Chamfer `27.2%`, held-out posed-vertex RMSE
`24.4%`, mean IoU, and boundary over a matched control. Its pooled-normal gain
was only `1.439°`, below the fixed `2°` gate. Exact next-step replay failed in
all arms, and the registered corrupted-track arm produced worse canonical
Chamfer than control.

E5 closed at this first scientific gate. No pretrained tracker was downloaded,
and no private RGB tracking, real training, development evaluation, SDF bridge,
or sealed access followed.

### E6 — interface-conforming field

The exact-predicate public field constructor passed ten valid fixtures and six
negative fixtures. It produced deterministic, watertight, oriented Euler-2
zero sets with zero sign mismatches over 200,000 probes and no non-interface
all-zero simplex, including near-contact gaps down to `0.1` reference cell.

The next registered prerequisite audited the unchanged accepted explicit
carrier. It has 27,554 finite vertices and 55,104 nondegenerate faces, zero
non-two-manifold edges, one component, consistent winding, watertight Euler-2
topology, and positive volume. Exact-predicate global testing nevertheless
found a self-intersection. E6 therefore closed without repairing the carrier,
constructing the private field, running distance/Eikonal probes, or opening
development evidence.

### P0 — exact next-step replay

The complete version-2 checkpoint transition passed bitwise next-step replay
on CPU and one L40S at steps 1, 7, 23, after topology projection, and after a
stage/mesh refresh. Negative controls with equal weights but mismatched RNG or
cursor failed as required. Legacy checkpoints remain readable but cannot claim
this proof.

### E7 — collision-preserving canonical optimization

The exact source prerequisite audited scaffold, shared initialization,
canonical/cage checkpoints, and SDF-projected carriers. Every eligible
same-connectivity source had exact self-intersections, so no source was legal.
E7 closed before repair, IPC barrier/CCD work, optimizer steps, or evaluation.

### E8 — opaque visibility training

The public opaque renderer gate passed every registered invariant and improved
known-shape Chamfer `94.11%` over the soft-splat control. In the paired real
smoke, control reduced its fixed objective `21.15%` and replayed exactly;
treatment reduced it `5.05%` but failed exact replay. E8 closed without retry,
bounded comparison, or development opening.

### E9 — tracklet outlier process

The clean public fixture exceeded its Chamfer, motion, and normal observability
targets and replayed exactly. None of the preregistered lambda multipliers was
both robust to corrupted tracklets and no worse than the two controls. E9
closed before tracker binding, private RGB, GPU training, or development use.

### E10 — embedded carrier transfer

The public parameter gate passed and froze one Alpha Wrapping parameter pair.
The one permitted real-source construction produced two byte-identical meshes.
Independent exact audit found a valid, outward-oriented, watertight,
one-component surface with zero self-intersections and complete registered
containment. Its Euler number was `-4`, not the required `2`, so E10 closed at
the source gate. Transfer, pose checks, development evaluation, parameter
fallback, and topology repair were not run.

### P1 — opaque renderer determinism diagnostic

P1 exposes clip, raster and raster-derivative buffers, point coverage,
interpolated normals, both antialias outputs, and geometry/attribute/final-
parameter gradients. The macOS CPU forward contract passed 100 bitwise repeats
with off-centre intrinsics. On one L40S, all registered forward stages and the
interpolated-attribute gradient remained bitwise stable for 100 repeats. The
geometry gradient first differed on repeat index 1 by about `7.45e-9`, and the
same difference propagated to the final cage gradient. Checkpoint-v2 next-step
replay passed at steps 1, 7, 12, and 23. P1 therefore closed at its exact
bitwise gate. No specific primitive was blamed, the renderer was not replaced,
and E8 was not reopened.

### E11 — genus-controlled carrier public gate

E11 is the only registered next scientific experiment. Its public constructor
uses CGAL 6.2 exact constructions to form a deterministic expanded convex
envelope, which is sphere topology by construction rather than repaired after
the fact. Five procedural source classes include concavity, a genus-one torus,
and disconnected components. The macOS gate ran each constructor twice and all
outputs were byte-identical, exactly self-intersection-free, watertight,
single-component, outward-oriented, Euler-2, and strictly containing the
source samples.

This is only a topology-construction result. Convex envelopes may erase
important concavities and gaps, so no private source construction or geometry
fidelity claim is authorized. After two dependency-only failures, the clean
Modal CPU gate completed all five fixture classes. Each constructor ran twice
with byte-identical output, and every independent exact audit found zero self-
intersections, one component, watertight outward orientation, and Euler number
2. It performed no image loading, optimization, private input reads,
development evaluation, or sealed access.

A second gate then tested geometric fidelity on eight frozen public fixtures.
All convex sphere/ellipsoid, rigid-motion, scale, byte-repeat, and independent
exact-topology checks passed. The concave pocket exceeded the fixed volume and
feature-stratum thresholds. The near-contact hairpin was more decisive: the
convex envelope closed all three registered exterior gap probes, more than
doubled reference volume, and failed P95 distance and median-normal gates.
E11 therefore closed locally without cloud replication or private-source work.

### E12-E15, P2, and P3 — exact-safe nonconvex carrier research

E12 tested a pressure shrinkwrap from the exact E11 envelope. Convex controls
passed, but source containment failed on ellipsoid/pocket cases and adaptive
hairpin refinement introduced 23 exact intersections. E13 added uniform
refinement and moving-wrap/static-source collision filtering. It restored
containment on several fixtures, but global CCD steps locked and the floating
midpoint hairpin acquired 125 exact intersections.

E14 replaced midpoint arithmetic with a dyadic lattice. Exact audits passed at
all refinement levels, including a 10,592-face hairpin with zero intersections,
but a floating closest-point identity proxy exceeded its preregistered ceiling.
The separate P2 correctness gate then propagated exact parent-face and integer
barycentric provenance. P2 passed all eight fixtures, both repetitions, and 48
level audits, proving the retained refinement is exact-safe without rewriting
E14's failed record.

E15 combined the P2 start with IPC barrier Newton steps. Its single public run
exceeded the two-hour ceiling inside full-mesh Tight-Inclusion CCD on the second
hairpin repetition, so partial fixture outputs were not promoted. P3 then
tested a conservative static-near/global-far collision partition. All 1,024
analytic paths passed with zero false-safe results, but the hairpin retained
107,581 near candidates. Native motion had more near pairs than full swept
pairs; tangential motion reduced the set but still exceeded the 30-second
narrow-phase ceiling. P3 closed without retry or a later scientific lane.

P4 replaced per-candidate TOI root solves with the pinned Planar-DAT trust
region filter, while retaining complete Tight-Inclusion as the independent
path judge. Its only official public run stopped at the analytic corpus: 82 of
1,024 paths were rejected, all tangent cases at scales between `1e-6` and
about `0.00645`. Crossing, separating, and zero-motion rejections were zero.
Representative filtered paths retained a positive gap, so the result exposes a
fixed-absolute-tolerance oracle mismatch rather than proving intersection.
P4 closed without reaching a mesh fixture, retrying, or activating E16.

P5 then normalized each joint start/end AABB to unit maximum side before TI.
It passed all 1,024 known raw labels and all 1,024 filtered-path labels with
zero repeat differences, while reproducing the 82 old absolute-oracle
rejections. This repairs the public evaluator contract but does not reopen P4
or establish a mesh result.

P6 reached the exact 10,592-face hairpin but both Planar-DAT mechanisms timed
out just above the 30-second ceiling before the normalized oracle ran. No
endpoint or partial fixture result was promoted. The dense trust-region stage,
not the corrected path judge, is the current public bottleneck.

A post-closure stage profile found 194,312 candidates, while broad phase, safe
distances, and collision-state construction together took about 0.016 seconds.
The >30-second cost is specifically the mixed-stencil Planar-DAT reduction.

P7 replaced that reduction with the isotropic per-vertex filter. The filter was
fast in isolation, but restricted all 5,298 wrap vertices and retained only
2.36% of tangential motion. It therefore failed the unchanged 25% motion gate.

P8-P11 next separated performance from semantics. Small-stencil batched
Planar-DAT matched upstream exactly, but a full upstream reference exceeded
five minutes. Complete candidate materialization then exposed floating branch
disagreements, and isolated upstream queries disproved both disagreement-only
and consensus-sampling proxies. The final P11 kernel queried every one of the
194,312 candidates through a singleton upstream filter. Two canonical passes
were bitwise identical and took `4.03/3.97` seconds, with zero singleton
failures. Nevertheless normalized full-path verification timed out and the
independent exact endpoint audit found 317 self-intersection pairs. P11 closes
the Planar-DAT family on global safety rather than local speed.

All of these gates used public procedural geometry. Private source reads,
development evaluations, image optimization, sealed access, and automatic
retries were zero.

### E16-E21 and B1 terminal public sequence

G16/E16 stopped when exact fine ambient-scaffold construction exceeded its
fixed 7,200-second public ceiling. E17's coarse bi-Lipschitz fallback retained
only `0.0162563` of tangential motion against the unchanged `0.25` gate. No
image reconstruction or private stage became eligible.

The isolated B1 SelfRecon prerequisite verified the pinned source and native
configuration, but the separately licensed assets, example data, and required
Linux/CUDA runtime were not bound. No baseline training ran and no substitute
asset was accepted.

E19's one-block exact determinant artifact retained `0.0746326` tangential
motion; its official JSON report failed and was not rerun. E20's four repeated
residual blocks retained `0.0931415`. E21 changed only the direction to follow
active determinant tangents. Native motion retained `0.368984` and passed its
independent exact endpoint/nesting audit with zero intersections. Tangential
motion improved to `0.136596`, but still failed `0.25`; every executed path,
area, topology, KKT, and tangent-residual gate passed before that stop.

The coarse fixed-connectivity ambient-map family is closed under E19-E21. No
grid, block-count, active-ratio, tolerance, certificate-floor, or time variant
is eligible under those IDs. E18 remains blocked. Protected reads, GPU/cloud
execution, spending, automatic retries, and sealed-test access were all zero.

## Reproduction boundary

The public repository can verify reusable source contracts and synthetic
geometry tests. It cannot reproduce the private subject result because the
media, evidence, SMPL assets, model weights, checkpoints, meshes, and exact
run provenance are intentionally private.

For public code review, the important facts are the declared data roles,
unchanged gates, experiment decisions above, source implementation, and tests.
Requests for a private artifact should be treated as unavailable rather than
filled with a proxy.

### E22/G22, E23, and B2 reconstruction-first record

E22/G22 optimized the bounded nonzero scalar values of a fixed tetrahedral
domain while rendering the actual shared-edge PL zero set. Its matched explicit
control used the same public views, renderer, image evidence, sampling,
optimizer budget, and initial surface. Target geometry remained evaluator-only.

The reusable core exposes differentiable shared-edge zero crossings, fixed
outer-boundary/sign validation, deterministic state hashing, conventional mesh
audit, and outward-rounded interval subdivision for scalar and explicit surface
paths. The complete G22 runner executed once. Both optimizer arms and the
endpoint artifact completed, after which report assembly failed on a relative-
versus-absolute path conversion. The official result is FAIL and the zero-retry
rule was enforced.

A read-only retained-endpoint audit reran no optimizer and could not promote
the artifact. It found a `76.99%` relative bidirectional-mean-distance
improvement and treatment held-out IoU `0.9265`, with passing endpoint topology,
probes, and exact embeddedness. The treatment nevertheless missed the inherited
`5°` directional-normal gate at about `17.87°` and `18.54°`; the control also
missed the `0.85` held-out-IoU gate. E22 is closed.

The full P2 regression retained all 10,592 faces and exact parent provenance.
Its three historical E11 hairpin exterior probes were all inside the inherited
envelope; this known closed-gap classification was preserved rather than
misreported as gap fidelity.

E23 changed only the trainable coordinates from direct vertices to the
invertible full-rank map `U=(M+L)V`; both arms retained all 1,662 scalar
geometric degrees of freedom. The matrix was positive definite and full rank,
with condition number `64.06`, round-trip and solve residuals below `1e-12`, and
a finite image-only gradient matching central differences. The direct arm then
completed 19 steps and the intrinsic arm 35 before each exhausted 32 complete-
path backtracks and rejected a step at scale zero. Their last accepted
endpoints passed topology, probes, and independent EPECK audits, but neither
reached replay step 37. The official 300-step comparison was not run. E23 is
closed without a coordinate, learning-rate, or remeshing variant; E24 remains
blocked because no valid representation was frozen.

B2 was registered only as an independent operational baseline workstream. It had to
bind the pinned upstream source, genuine licensed assets, official public
sequence, and compatible legacy Linux/CUDA environment before one complete
200-epoch reconstruction. No substitute asset, abbreviated run, render-only
path, private-first adapter, or output-file existence may count as success.
The source, genuine male model, and official 689-frame RGB/mask/normal example
are now bound in ignored storage with aligned frame identities and no extracted
pretrained endpoint. The Linux/CUDA runtime and compiled extensions remain
pending. The first remote image construction stopped before GPU execution when
the operating-system timezone package requested interactive input. The
build-only definition now fixes noninteractive UTC installation; this changed
no scientific source or dependency version. A second pre-GPU build passed that
layer and the pinned environment solve, then exposed two historical package-
build assumptions: OpenMesh required a system CMake executable and
pycocotools required its contemporary Cython build API. The third build-only
definition adds CMake and constrains PEP-517 build isolation to the compatible
Cython release. It does not change the runtime package set, upstream source,
inputs, probe, official command, or resource caps. Dependency GPU attempts,
training attempts, completed epochs, and automatic retries are still zero. The
third image then completed all builds and import checks, but app registration
stopped before worker dispatch because the conditional training function's
128-GiB ephemeral-disk request is below the platform's current 512-GiB
minimum. The deployment-only correction raises that request to the minimum;
the scientific stack and single-attempt contract were unchanged. The one L40S
dependency worker then started. Before the probe body or attempt marker could
run, the image's global path selected Python 3.8 for Modal's own runner, which
requires Python 3.10 or newer. The worker exited immediately. Under the frozen
zero-retry rule, B2 is closed; no 200-epoch training or private adapter ran.

## E25/B3 registration

E25 was the final registered native lane. It changed the coarse fixed-sign/facet-
normal representation to a multiresolution image-active SDF with continuous
gradient normals and differentiable FlexiCubes extraction. Its public fixture
sequence, stage-boundary exact audits, nonpromotable search intermediates,
post-commit connectivity freeze, `5°` normal gate, `10%` truth-distance
benefit gate, image nonregression gates, replay, and zero-retry cap are frozen
before implementation. No protected evidence is eligible.

B3 is independently registered to test a dual-runtime SelfRecon boundary. The
Python 3.11 control plane must remain outside the Python 3.8 legacy prefix and
invoke it only by absolute child path. This is a new baseline ID and does not
alter or rerun B2. One dependency attempt may enable one complete public
200-epoch schedule; no shortened or pretrained substitute is accepted.

## E25 implementation, preflight, and image checkpoint

E25's publishable representation layer now covers deterministic visual-hull
initialization, multiresolution neural SDF encoding, NeuS-style rendering,
continuous SDF-gradient normals, inverse-transpose transport, bilateral
integrability, pinned FlexiCubes extraction, stage commitment/connectivity
freeze, checkpoint replay, fixtures, and protected-read rejection.

The one local preflight passed 15 fixture-stage audits and five final
commitments, including the full 10,592-face P2 regression, exact endpoint
audits, derivative/transport controls, negative modalities, replay, and all six
read guards. It used no GPU or protected evidence. The first non-GPU Modal
image build failed before deployment because Ubuntu 22.04 supplied CGAL 5.4
while the retained exact auditor requires CGAL 6.2. The build-only correction
pinning the official CGAL 6.2 release then passed digest, compilation,
executable, and import checks. No GPU worker or experiment attempt started.

## E25 terminal public result

The sole registered E25 L40S worker claimed its exclusive attempt and failed
before training. Public target geometry was on CUDA while the retained camera
intrinsics were on CPU, so projection raised a mixed-device error. Zero
optimizer steps completed, no endpoint exists, and no partial result is
promotable. The zero-retry rule closes E25 without a device-placement rerun.
Private E25 and E26 are ineligible; protected and sealed reads remain zero.

## Resume state

B3's corrected non-GPU deployment passed the pinned legacy import assertion,
the separate control-plane import assertion, and the global-path isolation
assertion. Its isolated input was then populated only from the verified B2
public binding and passed a complete 2,759-file bytewise round trip. The sole
A10G dependency call nevertheless failed while remote GPU containers imported
the function module: a repository-relative build-constraint path did not exist
at the mounted location. The function body and scientific probe never started,
the output remained empty, and training attempts and completed epochs are zero.
The app was stopped and no automatic retry occurred. B3 is terminally closed.

E18/E24/E26 remain blocked. B4 D3-Human's one source-only audit passed its
pinned-source and official-command/example checks but failed project and
bundled-code licensing, SMPL-X binding, public-data terms, and evaluator
mapping. No data/model/GPU work occurred; B4 is closed. B5 REC-MV is eligible
only for its own separate source prerequisite registration at this checkpoint;
the terminal result is recorded below.
Preserve all historical metrics and closed-lane decisions, and never open the
sealed test.

### B5 source-only registration

B5 REC-MV is registered for one local five-minute source/access/license audit,
not for reconstruction. Official commit `5898020` and the full 200-epoch
`female-3-casual` self-rotation command/configuration are frozen. The audit must
resolve the repository's MIT-license versus Apache-badge ambiguity and bind the
exact PeopleSnapshot data, labeled garment templates, licensed SMPL model,
preprocessing licenses, and evaluator mapping. It allows zero downloads,
dependency builds, GPU-hours, spend, or retries. Any missing prerequisite
closes B5 before execution.

### B5 terminal prerequisite

The one audit verified the pinned REC-MV source, root MIT-license file,
official command, and full 200-epoch schedule. It failed the unambiguous-
license, public-data hash/terms, labeled-template hash/terms, licensed-SMPL,
preprocessing-license, and evaluator-mapping gates. No data/model download,
dependency build, GPU, training, private/development/sealed read, or retry
occurred. B5 and the serial B1-B5 external-baseline queue are closed.

### 0902 roadmap terminal state

E25 and B3-B5 each reached its frozen stop rule. E25 supplied no certified
representation, so private E25 and E26 never became eligible. No registered
lane is active; E18/E24/E26 remain blocked, all earlier results are preserved,
and the sealed test was not reopened.

## V2 handoff: G02 closed and a visual baseline frozen

Post-V1 V2 work established that the camera is fixed while the person turns,
qualified a fixed-camera human phase, preserved semantic and uncertainty
evidence, and accepted 249 bounded local material tracks. These are reusable
prerequisites, not a reconstruction endpoint.

The shortcut-resistant direct-field route passed engineering qualification and
ran one 600-step scientific attempt. Independent scoring found that it improved
boundary and normal direction relative to its matched control but reduced
held-out silhouette IoU by about `0.06305`. Its unmodified zero set also failed
the one-component/Euler-two body policy with eight components and Euler number
eight. G02 is terminal; its downstream normal, layered, and multimodal stages
were never activated. No authoritative layered result exists.

A deterministic zero-optimizer posed renderer now exposes the frozen V1 visual
baseline across 180 frames. Training/held-out hard-raster silhouette IoU is
about `0.579/0.583`. The replay follows the rough pose and turn but remains a
coarse, thin, untextured mesh rather than a realistic person. Source imagery
and the local comparison video are intentionally absent from this mirror.

For the weaker video-realism objective, G03 static appearance stopped before
development scoring after missing its training SSIM screen. The separately
registered G04 phase-conditioned appearance passed a target-excluded
144-frame training evaluation and the single frozen 36-frame development
evaluation. Development RGB was evaluator-only. Relative to neutral, held-out
RGB MAE improved `26.43%` and crop SSIM improved `0.05761`; it also beat G03 on
both metrics with identical foreground. The chronological 180-frame replay is
an automated passing, sequence-specific presentation result. Its independent
blinded human preference gate is still pending. Do not tune it on development,
reopen G02, use sealed evidence, or claim that the video satisfies layered
geometry.

## V2 handoff: D03 terminal and L03 pre-CUDA

D01 stopped before training evaluation when its real train topology precheck
failed. D02 passed train image metrics but failed exact self-intersection. D03
then built an independent capsule-tree implicit body. The closed body passed
one-component, Euler-2, outward, watertight, exact-intersection, replay, and
relative non-regression checks. Its one frozen development evaluation failed
the absolute `0.85` IoU and `0.015` boundary gates at `0.67428` and `0.015299`.
D03 is terminal and its body is not a clothed result.

Historically, L03 was separately registered and used that body only as a
prior-derived inner surface. Upper/lower clothing semantics are supported on all 144 training
frames and all 12 phase bins. The deterministic initializer emits one open
upper component with one boundary loop and one open lower component with two
boundary loops. Exact audits find no layer self-intersection or body
intersection; 73 source edges are registered as inter-layer contact. Public
training and one four-record local MPS qualification activate both displacement
parameter groups, change both, preserve bounded outward motion and fixed
connectivity, and restore exactly. No development record, sealed record,
scientific marker, or paid worker was used. At that time, the only eligible next
execution was one separately capped, zero-retry L40S engineering qualification;
the reproduction-first correction now freezes that route.

## Reproduction-first project correction

The project owner restored the original unified outer-surface objective as the
primary track. The reference-class result is one canonical implicit surface
with skeletal and residual non-rigid deformation, not a separated garment or
cloth-physics solution. Explicit garment boundaries, a material atlas, rest
metric, contact ownership, and strain are deferred optional research.

All earlier V1/V2/V3 and external-baseline outcomes remain historical facts.
The correction does not retroactively pass a failed experiment or erase a stop
decision. A new official/reference reproduction requires a separately governed
runtime and contract, followed by an isolated project-video adapter and only
then a project outer-surface fit.

New capture is staged: bare torso may validate camera, phase, pose, SDF,
deformation, and replay; a close-fitting top is the first clothed case; loose
clothing is a later stress test. Bare torso is never garment evidence.

## R01 registration and local binding

R01 is registered with a new prebuilt dual-runtime mechanism. Its zero-GPU
local audit passed the pinned official source and terms, official public example
bindings, licensed SMPL binding, and exact 689-frame RGB/mask/normal alignment.
No private project video, development record, sealed record, download, camera,
GPU worker, training step, spend, or retry was used.

R01 remains at `built`. Its only blocker is the absence of a new
content-addressed runtime artifact and manifest. The frozen recipe excludes
licensed assets and datasets and forbids dynamic Modal layer assembly or B2/B3
image reuse. No artifact was built or pushed.

The one subsequently authorized private build passed source checkout,
restricted-context checks, recipe verification, builder setup, and private-
registry authentication. The hosted runner then exhausted disk during image
construction and terminated before provenance generation or artifact
publication. A post-failure registry check found no package. It used no GPU,
training, camera, project evidence, development evidence, or sealed evidence,
and made zero retries.

This is an infrastructure failure rather than a scientific rejection, but the
authorized build attempt is terminal. It must not be retried or modified in
place. The executable entry point was removed after closure to prevent an
accidental second dispatch; its historical source remains in private history.
Continuing requires a separately registered successor whose changed mechanism
is a disk-capacity preflight and sufficiently provisioned builder, followed by
fresh owner authorization. R01 remains `built`; R02, R03, runtime import,
device qualification, and official training have not started.
