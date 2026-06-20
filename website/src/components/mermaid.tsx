'use client';

import { useEffect, useId, useState } from 'react';
import { useTheme } from 'next-themes';

// Renders a Mermaid diagram on the client. Works with static export — the SVG
// is produced after hydration. `chart` is the raw mermaid source.
export function Mermaid({ chart }: { chart: string }) {
  const id = useId();
  const [svg, setSvg] = useState('');
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    let active = true;

    void (async () => {
      const mermaid = (await import('mermaid')).default;
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'loose',
        fontFamily: 'inherit',
        theme: resolvedTheme === 'dark' ? 'dark' : 'default',
      });
      try {
        const { svg } = await mermaid.render(
          `mermaid-${id.replace(/[^a-zA-Z0-9]/g, '')}`,
          chart,
        );
        if (active) setSvg(svg);
      } catch (err) {
        console.error('Mermaid render failed:', err);
      }
    })();

    return () => {
      active = false;
    };
  }, [chart, id, resolvedTheme]);

  return (
    <div
      className="my-6 flex justify-center overflow-x-auto rounded-xl border bg-fd-card p-4 [&_svg]:max-w-full"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
