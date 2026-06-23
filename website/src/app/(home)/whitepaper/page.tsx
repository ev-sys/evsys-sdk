const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export default function WhitepaperPage() {
  const pdf = `${basePath}/whitepaper.pdf`;
  return (
    <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-6 py-8">
      <div className="mb-4 flex items-center justify-between gap-4">
        <h1 className="text-2xl font-bold tracking-tight">Whitepaper</h1>
        <a
          href={pdf}
          className="rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors hover:bg-fd-accent"
        >
          Open / download PDF
        </a>
      </div>
      <iframe
        src={pdf}
        title="EvSys whitepaper"
        className="h-[82vh] w-full rounded-lg border bg-fd-card"
      />
      <p className="mt-3 text-sm text-fd-muted-foreground">
        Trouble viewing inline?{' '}
        <a className="underline" href={pdf}>
          Open the PDF directly
        </a>
        .
      </p>
    </main>
  );
}
