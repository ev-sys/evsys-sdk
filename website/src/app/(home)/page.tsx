import Link from 'next/link';
import { Mermaid } from '@/components/mermaid';

// The whitepaper's "whole system at a glance" - the overall structure.
const SYSTEM = `flowchart TB
    CFG["ExperimentConfig (YAML)<br/>the single canonical artifact"]

    subgraph ORG["① Experiment layer - the organizing unit"]
        direction TB
        E["Experiment.run()"]
        EXP["expand: run / runs / matrix → arms<br/>n_repeats → seeded groups"]
        AR["ArmResult per run"]
        ER["ExperimentResult<br/>best_arm · best_score · conclusion"]
        E --> EXP --> AR --> ER
    end

    subgraph RUN["per-arm RunConfig (one training run)"]
        direction LR
        subgraph DATA["② Data surface"]
            direction TB
            SRC["raw source"] --> WSP["Workspace cache"] --> TRN["transforms[]"] --> TYP["typed rows"]
        end
        subgraph ALG["③ Algorithm surface"]
            direction TB
            BK["Backend<br/>mock · local · tinker"] --> AL["Algorithm.train(ctx)"]
            AL --> RES["RunResult<br/>status · metrics · artifacts"]
        end
        subgraph EVALS["④ Evaluation"]
            direction TB
            BMK["Benchmark (test, once)"]
            VAL["Validation (in-loop)"]
            MET["Metric · Verifier"]
        end
        TYP --> AL
        AL --> EVALS
    end

    subgraph OBS["⑤ Observability & storage"]
        direction LR
        LS["LogStore"] --- DC["DashboardClient"] --- ST["EvsysStore"]
    end

    REG["⑥ Registries (8) - kind → class<br/>algorithm · backend · transform · data_store<br/>log_store · metric · verifier · inference"]

    CFG --> E
    EXP --> RUN
    AR --> RES
    AR --> EVALS
    E --> OBS
    REG -. "resolves every 'kind:' in the YAML" .-> RUN`;

const AGENT_YAML = `# A coding agent launches an experiment by writing this - and
# sweeps, swaps algorithms, or registers new components by editing it.
matrix:
  axes:
    algorithm.kind: [sft, dpo, grpo]      # try three methods at once
    algorithm.params.lr: [1.0e-5, 2.0e-5]
base_run:
  data: { source: { kind: jsonl, params: { path: data/train.jsonl } } }
  backend: { kind: tinker }`;

function ComponentCard({
  href,
  index,
  title,
  children,
}: {
  href: string;
  index: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className="group rounded-xl border bg-fd-card p-5 text-left transition-colors hover:bg-fd-accent"
    >
      <div className="mb-1 flex items-center gap-2">
        <span className="text-xs font-semibold text-fd-primary">{index}</span>
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-fd-muted-foreground">{children}</p>
    </Link>
  );
}

export default function HomePage() {
  return (
    <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-6 py-16">
      {/* Hero */}
      <section className="flex flex-col items-center text-center">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`${process.env.NEXT_PUBLIC_BASE_PATH ?? ''}/logo.png`}
          alt=""
          width={132}
          height={132}
          className="mb-6 rounded-2xl"
        />
        <h1 className="text-5xl font-bold tracking-tight sm:text-7xl">
          evsys-sdk
        </h1>
        <p className="mt-6 max-w-3xl text-balance text-lg leading-relaxed text-fd-muted-foreground">
          Infrastructure for thousands of task-specialised models that learn
          continuously from every interaction.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/docs"
            className="rounded-lg bg-fd-primary px-5 py-2.5 text-sm font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
          >
            Get started
          </Link>
          <Link
            href="/docs/concepts/architecture"
            className="rounded-lg border px-5 py-2.5 text-sm font-medium transition-colors hover:bg-fd-accent"
          >
            How it works
          </Link>
          <Link
            href="/whitepaper"
            className="rounded-lg border px-5 py-2.5 text-sm font-medium transition-colors hover:bg-fd-accent"
          >
            Whitepaper
          </Link>
          <a
            href="https://github.com/ev-sys/evsys-sdk"
            className="rounded-lg border px-5 py-2.5 text-sm font-medium transition-colors hover:bg-fd-accent"
          >
            GitHub
          </a>
        </div>
      </section>

      {/* Main points */}
      <section className="mt-16 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ['Continually-learning models', 'Easily create continually-learning models on your own data.'],
          ['One declarative YAML', 'Every experiment is a single ExperimentConfig - nothing hidden in scripts.'],
          ['Autoresearch friendly', 'Make your coding agents fine-tune models on your own data on demand.'],
          ['Pluggable & continual', 'Eight registries for custom parts; weights chain so models keep learning.'],
        ].map(([t, d]) => (
          <div key={t} className="rounded-xl border bg-fd-card p-5">
            <h3 className="mb-1 font-semibold">{t}</h3>
            <p className="text-sm text-fd-muted-foreground">{d}</p>
          </div>
        ))}
      </section>

      {/* Main components */}
      <section className="mt-16">
        <h2 className="mb-2 text-2xl font-bold tracking-tight">
          The main components
        </h2>
        <p className="mb-6 text-fd-muted-foreground">
          One <code>ExperimentConfig</code> ties together five layers. Every{' '}
          <code>kind:</code> in it resolves through a registry.
        </p>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <ComponentCard index="①" title="Experiment" href="/docs/concepts/experiments">
            The organizing unit - a hypothesis, one or more runs, an
            auto-synthesized conclusion and <code>best_arm</code>.
          </ComponentCard>
          <ComponentCard index="②" title="Data surface" href="/docs/concepts/data">
            Raw sources → ordered <code>transforms</code> → standardized typed
            rows that carry only data.
          </ComponentCard>
          <ComponentCard index="③" title="Algorithm surface" href="/docs/concepts/algorithms">
            One contract - <code>train(ctx) -&gt; RunResult</code> - over any
            tinker-compatible backend.
          </ComponentCard>
          <ComponentCard index="④" title="Evaluation" href="/docs/concepts/algorithms">
            A test/validation firewall: <code>Benchmark</code> (once) vs{' '}
            <code>Validation</code> (in-loop), scored by metrics &amp; verifiers.
          </ComponentCard>
          <ComponentCard index="⑤" title="Plugins" href="/docs/concepts/plugins/algorithms">
            Eight registries - implement a protocol, register a{' '}
            <code>kind</code>, reference it in YAML.
          </ComponentCard>
          <ComponentCard index="⑥" title="API reference" href="/docs/evsys_sdk">
            263 pages auto-generated from the code - always in sync.
          </ComponentCard>
        </div>
      </section>

      {/* Overall structure diagram */}
      <section className="mt-16">
        <h2 className="mb-2 text-2xl font-bold tracking-tight">
          The overall structure
        </h2>
        <p className="mb-4 text-fd-muted-foreground">
          The whole system on one screen - one canonical config drives the
          Experiment layer, each run's data and algorithm surfaces, evaluation,
          and storage; the registries resolve every <code>kind:</code>.
        </p>
        <Mermaid chart={SYSTEM} />
      </section>

      {/* Built for coding agents - customisability */}
      <section className="mt-16">
        <h2 className="mb-2 text-2xl font-bold tracking-tight">
          Built for coding agents
        </h2>
        <p className="mb-5 max-w-3xl text-fd-muted-foreground">
          Because everything is one declarative artifact, a coding agent can
          drive the whole loop programmatically - and customise it at every
          layer without forking the library:
        </p>
        <div className="grid gap-6 lg:grid-cols-2">
          <ul className="flex flex-col gap-4 text-sm">
            <li className="rounded-xl border bg-fd-card p-4">
              <strong>Launch &amp; sweep.</strong> An agent edits the{' '}
              <code>ExperimentConfig</code> - flip an algorithm, add a{' '}
              <code>matrix</code> axis - and a whole campaign of runs expands
              from one file.
            </li>
            <li className="rounded-xl border bg-fd-card p-4">
              <strong>Register new parts.</strong> Every <code>kind:</code>{' '}
              resolves through a registry, so an agent can add a brand-new
              algorithm, verifier, or backend with{' '}
              <code>@register_algorithm(...)</code> - no SDK edit.
            </li>
            <li className="rounded-xl border bg-fd-card p-4">
              <strong>Learn from results.</strong> Outcomes are structured
              (<code>ExperimentResult</code> · <code>best_arm</code> ·{' '}
              <code>conclusion</code>), so an agent can read them and decide the
              next experiment.
            </li>
            <li className="rounded-xl border bg-fd-card p-4">
              <strong>Train continuously.</strong> Weights chain via{' '}
              <code>init_from_checkpoint</code>, so models keep learning across
              experiments instead of starting cold.
            </li>
          </ul>
          <div className="rounded-xl border bg-fd-card p-5">
            <div className="mb-2 text-xs font-medium text-fd-muted-foreground">
              what an agent edits
            </div>
            <pre className="overflow-x-auto text-left text-xs leading-relaxed">
              <code>{AGENT_YAML}</code>
            </pre>
          </div>
        </div>
        <div className="mt-6">
          <Link
            href="/docs/concepts/plugins/algorithms"
            className="text-sm font-medium text-fd-primary hover:underline"
          >
            Read how the registry pattern works →
          </Link>
        </div>
      </section>
    </main>
  );
}
