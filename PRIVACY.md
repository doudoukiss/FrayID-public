# Privacy policy for local reconstruction data

Human videos and reconstructed body geometry are sensitive biometric-adjacent
data. This repository does not include them and does not authorize publishing
them.

Keep the following outside Git and outside public artifact stores:

- source and reference videos, extracted frames, masks, normals, and keypoints;
- CameraHMR/SMPL outputs, body parameters, meshes, textures, and measurements;
- model weights, checkpoints, logs, reports, and generated previews;
- local manifests containing media hashes or identifying metadata;
- credentials, cloud volume names, account identifiers, and absolute user
  paths.

The supplied `.gitignore` blocks common media, geometry, model, output, and
credential formats. Treat it as a final guardrail, not as the primary privacy
control: it does not erase files from Git history. Inspect staged files before
every commit, publish only from a fresh allowlisted export, and keep the
canonical repository private.

Only process recordings from people who gave informed consent for the intended
research use. Define retention and deletion rules before collection, minimize
copies, restrict access, and review applicable biometric and data-protection law
for your jurisdiction.
