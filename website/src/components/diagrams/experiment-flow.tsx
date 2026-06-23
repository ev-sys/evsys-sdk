'use client';

import { useEffect, useState } from 'react';
import { useTheme } from 'next-themes';

// Clean SVG of the Experiment as the organizing unit:
// ExperimentConfig → Experiment (expands arms) → per-arm pipeline
// (Data → Training → Evaluation) → ExperimentResult. Same visual language as
// the intro diagram. Blue = user-authored, white = SDK pillar, grey = SDK output.

const LIGHT = {
  userFill: '#E8F0FE', userStroke: '#3B78E7', userText: '#1e3a8a',
  pillFill: '#ffffff', pillStroke: '#8A9099', pillText: '#111827',
  sdkFill: '#ECEEF1', sdkStroke: '#8A9099', sdkText: '#334155',
  edge: '#9aa0a6', group: '#c3c8cf',
  subText: '#64748b', labelBg: '#ffffff', labelText: '#475569',
};
const DARK = {
  userFill: '#16233b', userStroke: '#60a5fa', userText: '#dbeafe',
  pillFill: '#15191f', pillStroke: '#7c828b', pillText: '#f1f5f9',
  sdkFill: '#23272e', sdkStroke: '#7c828b', sdkText: '#e2e8f0',
  edge: '#6b7280', group: '#3a4049',
  subText: '#94a3b8', labelBg: '#0f1318', labelText: '#cbd5e1',
};
type Palette = typeof LIGHT;

function Node({
  c, x, y, w, h, fill, stroke, text, title, sub, bold = true,
}: {
  c: Palette; x: number; y: number; w: number; h: number;
  fill: string; stroke: string; text: string; title: string; sub?: string; bold?: boolean;
}) {
  const cx = x + w / 2;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={9} fill={fill} stroke={stroke} strokeWidth={1.6} />
      <text x={cx} y={sub ? y + h / 2 - 7 : y + h / 2} textAnchor="middle" dominantBaseline="middle" fontSize={15} fontWeight={bold ? 600 : 400} fill={text}>
        {title}
      </text>
      {sub && (
        <text x={cx} y={y + h / 2 + 11} textAnchor="middle" dominantBaseline="middle" fontSize={10.5} fill={c.subText}>
          {sub}
        </text>
      )}
    </g>
  );
}

function Label({ c, x, y, text }: { c: Palette; x: number; y: number; text: string }) {
  const w = text.length * 5.8 + 12;
  return (
    <g>
      <rect x={x - w / 2} y={y - 9} width={w} height={18} rx={4} fill={c.labelBg} />
      <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle" fontSize={10.5} fill={c.labelText}>{text}</text>
    </g>
  );
}

export function ExperimentFlow() {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const c = mounted && resolvedTheme === 'dark' ? DARK : LIGHT;
  const E = (d: string) => (
    <path d={d} fill="none" stroke={c.edge} strokeWidth={1.7} markerEnd="url(#ef-arr)" />
  );

  return (
    <div className="my-6 overflow-x-auto rounded-xl border bg-fd-card p-4">
      <svg viewBox="0 0 720 470" width="100%" style={{ minWidth: 520, maxWidth: 720, margin: '0 auto', display: 'block' }} fontFamily="inherit">
        <defs>
          <marker id="ef-arr" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">
            <path d="M0,0 L8,3.5 L0,7 Z" fill={c.edge} />
          </marker>
        </defs>

        {/* dashed "per arm" group around the pipeline */}
        <rect x={30} y={212} width={660} height={120} rx={12} fill="none" stroke={c.group} strokeWidth={1.3} strokeDasharray="5 5" />
        <text x={48} y={228} fontSize={11} fontStyle="italic" fill={c.subText}>per arm (RunConfig)</text>

        {/* edges */}
        {E('M360,66 V104')}
        {E('M360,160 V212')}
        {E('M225,287 H275')}
        {E('M445,287 H495')}
        {E('M580,322 V360 H360 V386')}

        <Label c={c} x={416} y={186} text="for each run" />
        <Label c={c} x={250} y={277} text="data format rows" />
        <Label c={c} x={470} y={277} text="checkpoint" />

        {/* nodes */}
        <Node c={c} x={240} y={18} w={240} h={48} fill={c.userFill} stroke={c.userStroke} text={c.userText} title="ExperimentConfig" sub="one YAML - the canonical artifact" />
        <Node c={c} x={210} y={104} w={300} h={56} fill={c.pillFill} stroke={c.pillStroke} text={c.pillText} title="Experiment" sub="run / runs / matrix → arms · n_repeats → seeds" />

        <Node c={c} x={55} y={252} w={170} h={70} fill={c.pillFill} stroke={c.pillStroke} text={c.pillText} title="Data" sub="raw → data format rows" />
        <Node c={c} x={275} y={252} w={170} h={70} fill={c.pillFill} stroke={c.pillStroke} text={c.pillText} title="Training" sub="build_batch()" />
        <Node c={c} x={495} y={252} w={170} h={70} fill={c.pillFill} stroke={c.pillStroke} text={c.pillText} title="Evaluation" sub="validation + benchmark" />

        <Node c={c} x={210} y={386} w={300} h={52} fill={c.sdkFill} stroke={c.sdkStroke} text={c.sdkText} title="ExperimentResult" sub="best_arm · conclusion · metrics" />
      </svg>
    </div>
  );
}
