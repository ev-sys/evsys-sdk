# evsys-sdk documentation site

The public docs for `evsys-sdk`, built with [Fumadocs](https://fumadocs.dev)
(Next.js, static export) and deployed to GitHub Pages at
**https://trajectory-ai.github.io/evsys-sdk/**.

## Local development

```bash
cd website
pnpm install
pnpm dev          # http://localhost:3000
```

## Content

| Source | What it is |
| --- | --- |
| `content/docs/*.mdx`, `*.md` | Hand-written guides (intro, installation, cookbook, design, reference) |
| `content/docs/(api)/evsys_sdk/**` | **Auto-generated** API reference (do not edit by hand) |

### Regenerating the API reference

The API pages are introspected straight from the installed `evsys_sdk`
package via [`fumadocs-python`](https://fumadocs.dev/docs/integrations/python).
Re-run this whenever the SDK's public surface changes:

```bash
pnpm gen:api      # fumapy-generate evsys_sdk -> JSON -> MDX under content/docs/(api)
```

This requires the SDK's venv with `fumadocs-python` installed:

```bash
uv pip install --python ../.venv/bin/python ./node_modules/fumadocs-python
```

## Deployment

The SDK repo is **private** (and GitHub Pages isn't available for private
repos on the free plan), so the built site is published to a separate **public**
repo — [`trajectory-ai/evsys-sdk-docs`](https://github.com/trajectory-ai/evsys-sdk-docs)
— whose GitHub Pages serves **https://trajectory-ai.github.io/evsys-sdk-docs/**.

To publish (refresh the API reference first if the SDK changed):

```bash
pnpm gen:api      # optional: re-introspect evsys_sdk
pnpm deploy       # build with PAGES_BASE_PATH=/evsys-sdk-docs and force-push out/ to the public repo
```

`pnpm deploy` runs `scripts/deploy.sh`. The `PAGES_BASE_PATH` env var sets the
project-pages sub-path; local `pnpm build` / `pnpm dev` leave it empty so the
site works at `localhost`.

> If the SDK repo is ever made public (or the org upgrades to a paid plan), you
> can deploy this `website/` directly via a GitHub Actions Pages workflow
> instead of the separate public repo.
