# Publishing to the Comfy Registry

Nothing in this repository publishes automatically. The workflow in
[`.github/workflows/publish.yml`](../.github/workflows/publish.yml) only runs when you start it by hand,
and it stops straight away if the `REGISTRY_ACCESS_TOKEN` secret is missing. Once a version is
on the registry, ComfyUI-Manager users can install it by name.

## One-time setup

1. **Create the publisher.** Sign in at <https://registry.comfy.org> with GitHub and create a
   publisher whose **ID is `somarawad`**. It must match `PublisherId` in `pyproject.toml`.
   The ID can't be changed later. If it is taken, pick another and update `pyproject.toml`.
2. **Create an API key.** On the publisher page, open *API Keys*, create a key (for example
   "GitHub Actions"), and copy it. It is shown only once.
3. **Store it as a repository secret.** On GitHub, go to *Settings → Secrets and variables →
   Actions → New repository secret*. Name it `REGISTRY_ACCESS_TOKEN` and paste the key.

## Each release

1. Bump `version` in `pyproject.toml`. The registry refuses a version that already exists, and
   `comfy node publish` uploads exactly that version. Add the release to `CHANGELOG.md`.
2. Check the package locally. [comfy-cli](https://github.com/Comfy-Org/comfy-cli) is what the
   GitHub Action runs:

   ```bash
   pip install comfy-cli
   comfy node validate     # pyproject fields + the registry's security checks
   comfy node pack         # builds node.zip exactly as it would be uploaded (honours .comfyignore)
   ```

3. Merge the release PR into `main`, and wait for the `tests` workflow to pass.
4. Publish: open *Actions → Publish to Comfy Registry → Run workflow*, choose `main`, and type
   the version you are publishing. The job checks that it matches `pyproject.toml`, then runs
   [`Comfy-Org/publish-node-action@1.0.1`](https://github.com/Comfy-Org/publish-node-action).
5. Check the node page on <https://registry.comfy.org> and install it once through
   ComfyUI-Manager (or `comfy node install comfyui-shotplanner`) to confirm.

You can also publish without GitHub Actions. From a clean checkout of `main`, run
`comfy node publish --token <key>`.

## What gets uploaded

`comfy node pack` and `comfy node publish` zip the git-tracked files, minus those listed in
[`.comfyignore`](../.comfyignore). The treatment sample renders, screenshots, CI files and dev
tools stay on GitHub only, so ComfyUI users download a small package.

## Registry fields (`pyproject.toml`)

| Field | Value | Notes |
|---|---|---|
| `[project] name` | `comfyui-shotplanner` | The node's ID on the registry. Never change it after the first publish. |
| `version` | `1.1.0` | Semantic version; bump it for every publish. |
| `license` | `{ file = "LICENSE" }` | MIT. |
| `dependencies` | `[]` | The pack only needs torch, numpy and Pillow, which ComfyUI already installs. |
| `[project.urls]` | Repository, Documentation, Bug Tracker, Changelog | Shown on the registry page. |
| `[tool.comfy] PublisherId` | `somarawad` | Must match the publisher you created. |
| `DisplayName` | `Shot Planner` | |
| `Icon` | `docs/img/icon.png` on `main` (raw URL) | 400×400 PNG. The URL resolves once this release is merged. |
