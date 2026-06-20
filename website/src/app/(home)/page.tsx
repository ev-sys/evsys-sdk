import Link from 'next/link';

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center px-6 text-center">
      <span className="mb-4 rounded-full border px-3 py-1 text-xs font-medium text-fd-muted-foreground">
        v0.1.0 · declarative LLM training
      </span>
      <h1 className="max-w-3xl text-balance text-4xl font-bold tracking-tight sm:text-6xl">
        Run LLM experiments from a single{' '}
        <span className="text-fd-primary">YAML</span>.
      </h1>
      <p className="mt-6 max-w-2xl text-balance text-lg text-fd-muted-foreground">
        <code className="font-semibold">evsys-sdk</code> is a modular framework
        for SFT, RL, and distillation — pluggable algorithms, verifiers,
        metrics, and backends. Local on TRL, remote on Tinker.
      </p>
      <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
        <Link
          href="/docs"
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
      <pre className="mt-12 w-full max-w-xl overflow-x-auto rounded-xl border bg-fd-card p-5 text-left text-sm">
        <code>{`uv pip install -e ".[tinker,local]"
evsys validate config.yaml --deep
evsys run config.yaml`}</code>
      </pre>
    </main>
  );
}
