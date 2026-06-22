import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

// Set PAGES_BASE_PATH=/evsys-sdk at build time so assets resolve under the
// GitHub Pages project URL. Left empty for local dev/preview so the site works
// at http://localhost.
const basePath = process.env.PAGES_BASE_PATH || '';

// Static export is only needed for the production build that ships to GitHub
// Pages. In `next dev` it breaks client-side RSC navigation (links fall back to
// slow full-page reloads), so enable it for production builds only.
const isDev = process.env.NODE_ENV === 'development';

/** @type {import('next').NextConfig} */
const config = {
  ...(isDev ? {} : { output: 'export', trailingSlash: true }),
  reactStrictMode: true,
  basePath,
  images: { unoptimized: true },
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default withMDX(config);
