import { rimraf } from 'rimraf';
import * as Python from 'fumadocs-python';
import * as fs from 'node:fs/promises';

// JSON produced by `fumapy-generate evsys_sdk --dir website`
const jsonPath = './evsys_sdk.json';

async function generate() {
  // Nest under `evsys_sdk` so the generated URLs (/docs/evsys_sdk/...) match
  // the file layout. `(api)` is a route group, so it adds no URL segment.
  const out = 'content/docs/(api)/evsys_sdk';
  // clean previous output
  await rimraf('content/docs/(api)');

  const content = JSON.parse((await fs.readFile(jsonPath)).toString());
  const converted = Python.convert(content, {
    baseUrl: '/docs',
  });

  await Python.write(converted, {
    outDir: out,
  });

  console.log(`Wrote ${converted.length ?? 'API'} pages to ${out}`);
}

void generate();
