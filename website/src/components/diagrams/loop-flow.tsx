'use client';

import { useEffect, useId, useState } from 'react';
import { useTheme } from 'next-themes';

// Closed continual-learning loop: production traces → decision agent →
// Autoresearch → deployment evals → deploy → back. Autoresearch stays
// highlighted; no phase/released badge.

const LIGHT = {
  pillFill: '#ffffff',
  pillStroke: '#c3c8cf',
  pillText: '#111827',
  accentFill: '#E8F0FE',
  accentStroke: '#3B78E7',
  accentText: '#1d4ed8',
  edge: '#9aa0a6',
  subText: '#64748b',
};

const DARK = {
  pillFill: '#15191f',
  pillStroke: '#3a4049',
  pillText: '#f1f5f9',
  accentFill: '#16233b',
  accentStroke: '#60a5fa',
  accentText: '#bfdbfe',
  edge: '#6b7280',
  subText: '#94a3b8',
};

type Palette = typeof LIGHT;

type Box = {
  x: number;
  label: string;
  sub?: string;
  accent?: boolean;
};

export function LoopFlow() {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const c: Palette = mounted && resolvedTheme === 'dark' ? DARK : LIGHT;
  const markerId = useId().replace(/:/g, '');

  const w = 168;
  const h = 58;
  const y = 30;
  const midY = y + h / 2;
  const boxes: Box[] = [
    { x: 8, label: 'Production traces' },
    { x: 216, label: 'Agent: train on this?' },
    { x: 424, label: 'Autoresearch', accent: true },
    { x: 632, label: 'Deployment evals', sub: 'gate the checkpoint' },
    { x: 840, label: 'Deploy', sub: 'if it improves' },
  ];
  const loopPath = `M ${boxes[4].x + w / 2} ${y + h} L ${boxes[4].x + w / 2} 158 L ${boxes[0].x + w / 2} 158 L ${boxes[0].x + w / 2} ${y + h + 3}`;

  return (
    <div className="my-6 overflow-x-auto rounded-xl border bg-fd-card p-4 sm:p-5">
      <style>{`
        @keyframes evsys-loop-flow {
          to { stroke-dashoffset: -24; }
        }
        @keyframes evsys-loop-feedback {
          to { stroke-dashoffset: -32; }
        }
        @keyframes evsys-loop-accent {
          0%, 100% { opacity: 0.92; }
          50% { opacity: 1; }
        }
        .evsys-loop-flow {
          stroke-dasharray: 6 10;
          animation: evsys-loop-flow 1.8s linear infinite;
        }
        .evsys-loop-feedback {
          stroke-dasharray: 5 7;
          animation: evsys-loop-feedback 2.6s linear infinite;
        }
        .evsys-loop-accent {
          animation: evsys-loop-accent 3.2s ease-in-out infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .evsys-loop-flow, .evsys-loop-feedback, .evsys-loop-accent { animation: none; }
        }
      `}</style>
      <svg
        viewBox="0 0 1016 196"
        width="100%"
        style={{ minWidth: 720, maxWidth: 1016, margin: '0 auto', display: 'block' }}
        fontFamily="inherit"
        role="img"
        aria-label="The continual-learning loop: production traces, decision agent, autoresearch, deployment evals, deploy, and back."
      >
        <defs>
          <marker
            id={`loop-arrow-${markerId}`}
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="6.5"
            markerHeight="6.5"
            orient="auto-start-reverse"
          >
            <path
              d="M1 1.5 L8.5 5 L1 8.5"
              fill="none"
              stroke={c.edge}
              strokeWidth={1.6}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </marker>
        </defs>

        {[0, 1, 2, 3].map((i) => {
          const x1 = boxes[i].x + w;
          const x2 = boxes[i + 1].x - 3;
          return (
            <g key={i}>
              <line
                x1={x1}
                y1={midY}
                x2={x2}
                y2={midY}
                stroke={c.edge}
                strokeOpacity={0.35}
                strokeWidth={1.4}
              />
              <line
                className="evsys-loop-flow"
                x1={x1}
                y1={midY}
                x2={x2}
                y2={midY}
                stroke={c.edge}
                strokeOpacity={0.85}
                strokeWidth={1.4}
                strokeLinecap="round"
                markerEnd={`url(#loop-arrow-${markerId})`}
              />
            </g>
          );
        })}

        <text
          x={(boxes[1].x + w + boxes[2].x) / 2}
          y={midY - 8}
          textAnchor="middle"
          fontSize={11}
          fill={c.subText}
        >
          yes
        </text>

        <path
          d={loopPath}
          fill="none"
          stroke={c.edge}
          strokeOpacity={0.28}
          strokeWidth={1.4}
        />
        <path
          className="evsys-loop-feedback"
          d={loopPath}
          fill="none"
          stroke={c.edge}
          strokeOpacity={0.8}
          strokeWidth={1.4}
          strokeLinecap="round"
          markerEnd={`url(#loop-arrow-${markerId})`}
        />
        <text
          x={(boxes[0].x + boxes[4].x + w) / 2}
          y={176}
          textAnchor="middle"
          fontSize={11}
          fill={c.subText}
        >
          deployed model serves users → new traces
        </text>

        {boxes.map((b) => (
          <g key={b.label} className={b.accent ? 'evsys-loop-accent' : undefined}>
            <rect
              x={b.x}
              y={y}
              width={w}
              height={h}
              rx={11}
              fill={b.accent ? c.accentFill : c.pillFill}
              stroke={b.accent ? c.accentStroke : c.pillStroke}
              strokeWidth={b.accent ? 1.8 : 1.3}
            />
            <text
              x={b.x + w / 2}
              y={b.sub ? y + 25 : y + h / 2 + 4.5}
              textAnchor="middle"
              fontSize={13}
              fontWeight={600}
              fill={b.accent ? c.accentText : c.pillText}
            >
              {b.label}
            </text>
            {b.sub && (
              <text
                x={b.x + w / 2}
                y={y + 42}
                textAnchor="middle"
                fontSize={10.5}
                fill={c.subText}
              >
                {b.sub}
              </text>
            )}
          </g>
        ))}
      </svg>
      <p className="mt-3 text-center text-xs text-fd-muted-foreground">
        The closed loop. The blue{' '}
        <span className="font-medium text-fd-primary">autoresearch</span> stage
        finds a better checkpoint before deploy.
      </p>
    </div>
  );
}
