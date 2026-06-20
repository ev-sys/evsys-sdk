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

`.github/workflows/deploy-docs.yml` builds and publishes on every push to
`dev` that touches `website/`. The build sets `PAGES_BASE_PATH=/evsys-sdk`
so assets resolve under the project-pages sub-path; local builds leave it empty.
