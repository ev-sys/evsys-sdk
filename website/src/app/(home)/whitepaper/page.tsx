const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

// Just the PDF viewer — fill the viewport below the nav.
export default function WhitepaperPage() {
  return (
    <iframe
      src={`${basePath}/whitepaper.pdf`}
      title="EvSys whitepaper"
      className="w-full border-0"
      style={{ height: 'calc(100dvh - 56px)' }}
    />
  );
}
