const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

// The whitepaper, rendered as page images (scrollable). PDFs in an <iframe>
// only show page 1 on iOS Safari — stacked images page on every device, and
// pinch-zoom works. Regenerate with:
//   pdftocairo -png -r 170 docs/whitepaper/overview.pdf website/public/whitepaper/page
const PAGES = 7;

export default function WhitepaperPage() {
  const pages = Array.from({ length: PAGES }, (_, i) => i + 1);
  return (
    <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-4 sm:py-8">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h1 className="text-lg font-semibold tracking-tight">Whitepaper</h1>
        <a
          href={`${basePath}/whitepaper.pdf`}
          className="rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors hover:bg-fd-accent"
        >
          Download PDF
        </a>
      </div>
      <div className="flex flex-col gap-3 sm:gap-4">
        {pages.map((n) => (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={n}
            src={`${basePath}/whitepaper/page-${n}.png`}
            alt={`Whitepaper page ${n}`}
            width={1406}
            height={1988}
            loading={n <= 2 ? 'eager' : 'lazy'}
            className="h-auto w-full rounded-lg border bg-white shadow-sm"
          />
        ))}
      </div>
    </main>
  );
}
