import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import { Step, Steps } from 'fumadocs-ui/components/steps';
import * as Python from 'fumadocs-python/components';
import { Mermaid } from './mermaid';
import { TopLevelFlow } from './diagrams/top-level-flow';
import { ExperimentFlow } from './diagrams/experiment-flow';
import { LoopFlow } from './diagrams/loop-flow';

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    Step,
    Steps,
    Mermaid,
    TopLevelFlow,
    ExperimentFlow,
    LoopFlow,
    ...Python,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
