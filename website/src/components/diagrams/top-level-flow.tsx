'use client';

import { useEffect, useState } from 'react';
import { useTheme } from 'next-themes';

// A clean version of the whitepaper's top-level dataflow: the three pillars
// Data → Training → Evaluation, with a coding agent editing the Data and
// Training (Algorithm) surfaces. Blue = user/agent-authored, white = SDK pillar.

const LIGHT = {
  pillFill: '#ffffff', pillStroke: '#8A9099', pillText: '#111827',
  agentFill: '#E8F0FE', agentStroke: '#3B78E7', agentText: '#1d4ed8',
  edge: '#9aa0a6', edgeBlue: '#3B78E7',
  subText: '#64748b', labelBg: '#ffffff', labelText: '#475569',
};

const DARK = {
  pillFill: '#15191f', pillStroke: '#7c828b', pillText: '#f1f5f9',
  agentFill: '#16233b', agentStroke: '#60a5fa', agentText: '#bfdbfe',
  edge: '#6b7280', edgeBlue: '#60a5fa',
  subText: '#94a3b8', labelBg: '#0f1318', labelText: '#cbd5e1',
};

type Palette = typeof LIGHT;

function Pillar({
  c, x, title, sub,
}: {
  c: Palette; x: number; title: string; sub: string;
}) {
  const w = 180;
  const h = 92;
  const y = 70;
  const cx = x + w / 2;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={10} fill={c.pillFill} stroke={c.pillStroke} strokeWidth={1.6} />
      <text x={cx} y={y + h / 2 - 7} textAnchor="middle" dominantBaseline="middle" fontSize={16} fontWeight={600} fill={c.pillText}>
        {title}
      </text>
      <text x={cx} y={y + h / 2 + 13} textAnchor="middle" dominantBaseline="middle" fontSize={11} fill={c.subText}>
        {sub}
      </text>
    </g>
  );
}

function Label({ c, x, y, text, color }: { c: Palette; x: number; y: number; text: string; color?: string }) {
  const w = text.length * 5.8 + 12;
  return (
    <g>
      <rect x={x - w / 2} y={y - 9} width={w} height={18} rx={4} fill={c.labelBg} />
      <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle" fontSize={11} fill={color ?? c.labelText}>
        {text}
      </text>
    </g>
  );
}

export function TopLevelFlow() {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const c = mounted && resolvedTheme === 'dark' ? DARK : LIGHT;

  return (
    <div className="my-6 overflow-x-auto rounded-xl border bg-fd-card p-4">
      <svg
        viewBox="0 0 740 330"
        width="100%"
        style={{ minWidth: 480, maxWidth: 740, margin: '0 auto', display: 'block' }}
        fontFamily="inherit"
      >
        <defs>
          <marker id="tlf-g" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">
            <path d="M0,0 L8,3.5 L0,7 Z" fill={c.edge} />
          </marker>
          <marker id="tlf-b" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">
            <path d="M0,0 L8,3.5 L0,7 Z" fill={c.edgeBlue} />
          </marker>
        </defs>

        {/* pipeline: Data → Training → Evaluation */}
        <path d="M210,116 H280" fill="none" stroke={c.edge} strokeWidth={1.8} markerEnd="url(#tlf-g)" />
        <path d="M460,116 H530" fill="none" stroke={c.edge} strokeWidth={1.8} markerEnd="url(#tlf-g)" />

        {/* coding agent edits Data + Training (dotted blue) */}
        <path d="M240,250 C205,215 165,195 128,166" fill="none" stroke={c.edgeBlue} strokeWidth={1.6} strokeDasharray="2 4" markerEnd="url(#tlf-b)" />
        <path d="M280,250 C320,222 348,198 366,166" fill="none" stroke={c.edgeBlue} strokeWidth={1.6} strokeDasharray="2 4" markerEnd="url(#tlf-b)" />

        <Label c={c} x={168} y={206} text="edit" color={c.edgeBlue} />
        <Label c={c} x={330} y={206} text="edit" color={c.edgeBlue} />

        {/* pillars */}
        <Pillar c={c} x={30} title="Data" sub="raw → data format rows" />
        <Pillar c={c} x={280} title="Training" sub="build_batch()" />
        <Pillar c={c} x={530} title="Evaluation" sub="validation + benchmark" />

        {/* coding agent */}
        <g>
          <rect x={185} y={250} width={150} height={52} rx={10} fill={c.agentFill} stroke={c.agentStroke} strokeWidth={1.5} strokeDasharray="3 3" />
          <text x={260} y={276} textAnchor="middle" dominantBaseline="middle" fontSize={14} fontWeight={600} fontStyle="italic" fill={c.agentText}>
            Coding agent
          </text>
        </g>
      </svg>
    </div>
  );
}
