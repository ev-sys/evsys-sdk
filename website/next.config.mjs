import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

// Set PAGES_BASE_PATH=/trajectory-labs-sdk in CI so assets resolve under the
// GitHub Pages project URL (https://trajectory-ai.github.io/trajectory-labs-sdk/).
// Left empty for local dev/preview so the site works at http://localhost.
const basePath = process.env.PAGES_BASE_PATH || '';

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  reactStrictMode: true,
  basePath,
  // Export every route as a directory (docs/index.html) so GitHub Pages
  // resolves both /docs and /docs/ — Pages does directory-style lookups.
  trailingSlash: true,
  images: { unoptimized: true },
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default withMDX(config);
