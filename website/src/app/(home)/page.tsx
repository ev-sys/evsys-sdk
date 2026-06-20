import Link from 'next/link';
import { Mermaid } from '@/components/mermaid';

const PIPELINE = `flowchart LR
    YAML["config.yaml"] --> EXP["Experiment"]
    EXP -->|"expand arms"| RUNS["Run 1..N"]
    DATA[("Dataset")] --> ALG
    RUNS --> ALG{"Algorithm<br/>SFT · RL · Distill"}
    ALG -->|"rollouts"| BACK["Backend<br/>local · tinker · fireworks"]
    BACK --> EVAL["Eval<br/>verifiers + metrics"]
    EVAL --> ART[("Artifacts<br/>checkpoints · scores")]`;

const CONFIG = `name: sft-smoke-test
run:
  data:
    source: { kind: jsonl, params: { path: data/train.jsonl } }
    transforms: [{ kind: jsonl_to_chat }]
  model: { name: meta-llama/Llama-3.2-1B }
  algorithm: { kind: sft, params: { epochs: 1, lr: 2.0e-5 } }
  backend: { kind: mock }`;

function Card({
  href,
  title,
  children,
}: {
  href: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className="rounded-xl border bg-fd-card p-5 text-left transition-colors hover:bg-fd-accent"
    >
      <h3 className="mb-1 font-semibold">{title}</h3>
      <p className="text-sm text-fd-muted-foreground">{children}</p>
    </Link>
  );
}

export default function HomePage() {
  return (
    <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-6 py-16">
      {/* Hero */}
      <section className="flex flex-col items-center text-center">
        <span className="mb-4 rounded-full border px-3 py-1 text-xs font-medium text-fd-muted-foreground">
          v0.1.0 · SFT · RL · Distillation
        </span>
        <h1 className="max-w-3xl text-balance text-4xl font-bold tracking-tight sm:text-6xl">
          LLM training experiments from a single{' '}
          <span className="text-fd-primary">YAML</span>.
        </h1>
        <p className="mt-6 max-w-2xl text-balance text-lg text-fd-muted-foreground">
          <code className="font-semibold">evsys-sdk</code> is a declarative
          framework for SFT, RL, and distillation — pluggable algorithms,
          verifiers, metrics, and backends. No glue code, no training-script
          sprawl.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/docs/quickstart"
            className="rounded-lg bg-fd-primary px-5 py-2.5 text-sm font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
          >
            Get started
          </Link>
          <Link
            href="/docs/evsys_sdk"
            className="rounded-lg border px-5 py-2.5 text-sm font-medium transition-colors hover:bg-fd-accent"
          >
            API reference
          </Link>
          <a
            href="https://github.com/trajectory-ai/evsys-sdk"
            className="rounded-lg border px-5 py-2.5 text-sm font-medium transition-colors hover:bg-fd-accent"
          >
            GitHub
          </a>
        </div>
      </section>

      {/* Killer demo: one file + one command */}
      <section className="mt-16 grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border bg-fd-card p-5">
          <div className="mb-2 text-xs font-medium text-fd-muted-foreground">
            config.yaml
          </div>
          <pre className="overflow-x-auto text-left text-xs leading-relaxed">
            <code>{CONFIG}</code>
          </pre>
        </div>
        <div className="flex flex-col justify-center rounded-xl border bg-fd-card p-5">
          <div className="mb-2 text-xs font-medium text-fd-muted-foreground">
            run it
          </div>
          <pre className="overflow-x-auto text-left text-sm leading-relaxed">
            <code>{`$ evsys validate config.yaml --deep
$ evsys run config.yaml

✓ experiment sft-smoke-test
  arm 0  succeeded  → final_checkpoint`}</code>
          </pre>
          <p className="mt-4 text-sm text-fd-muted-foreground">
            The same shape runs locally or on hosted backends — you only swap{' '}
            <code>backend.kind</code>.
          </p>
        </div>
      </section>

      {/* Pipeline diagram */}
      <section className="mt-16">
        <h2 className="mb-1 text-center text-sm font-medium uppercase tracking-wide text-fd-muted-foreground">
          The spine
        </h2>
        <p className="mb-4 text-center text-fd-muted-foreground">
          config.yaml → Experiment → arms → backend → eval → artifacts
        </p>
        <Mermaid chart={PIPELINE} />
      </section>

      {/* Why */}
      <section className="mt-16 grid gap-4 sm:grid-cols-2">
        <Card href="/docs/concepts/architecture" title="One YAML drives everything">
          One file expands into a whole campaign of runs via <code>matrix</code>.
        </Card>
        <Card href="/docs/concepts/extensibility" title="Pluggable by design">
          Eight registries. Add a method with{' '}
          <code>@register_algorithm</code> — no fork.
        </Card>
        <Card href="/docs/concepts/algorithms" title="SFT · RL · distillation">
          One tool, one config shape, three training paradigms.
        </Card>
        <Card href="/docs/evsys_sdk" title="Auto-generated API reference">
          263 pages introspected from the code, always in sync.
        </Card>
      </section>
    </main>
  );
}
