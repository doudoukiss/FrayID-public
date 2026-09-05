# FrayID public contributor guide

Read `docs/PROJECT_CONTEXT.md` for the scientific target and
`configs/evaluation/current_project_track.yaml` for current public state.
Historical handoff and roadmap files are optional references, not mandatory
reading for routine changes.

- Use targeted local tests during development.
- Do not add or depend on GitHub Actions CI.
- Do not infer private artifacts from public filenames.
- Never add human media, body data, weights, checkpoints, meshes, outputs,
  credentials, private paths, run identifiers, or artifact fingerprints.
- The target is a unified canonical outer surface, not separated-garment or
  cloth-physics reconstruction.
