#!/usr/bin/env bash
# Build the docs site and publish it to the PUBLIC GitHub Pages repo
# (trajectory-ai/evsys-sdk-docs). The SDK repo itself stays private, so we
# publish only the built static output to a separate public repo.
#
# Refresh the API reference from the code first if the SDK changed:
#   pnpm gen:api
set -euo pipefail
cd "$(dirname "$0")/.."

PAGES_BASE_PATH=/evsys-sdk-docs pnpm build

cd out
touch .nojekyll            # let GitHub Pages serve the _next/ folder
rm -rf .git
git init -q -b main
git add -A
git commit -q -m "Deploy evsys-sdk docs"
git remote add origin https://github.com/trajectory-ai/evsys-sdk-docs.git
git push -f origin main
echo "Published → https://trajectory-ai.github.io/evsys-sdk-docs/"
