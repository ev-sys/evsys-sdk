'use client';

// Dependency-free inline-SVG charts for the continual-learning research page.
// No recharts — the site ships as a static export with a frozen lockfile, so we
// draw these by hand. Series colours are explicit; axis/grid use fd-* tokens via
// currentColor so they follow light/dark theme.

const SFT = '#94a3b8';
const HYBRID = '#7c3aed';
const MULTI = '#059669';
const PLAIN = '#f59e0b';
const FORGET = '#dc2626';

const W = 640;
const H = 300;
const PAD = { top: 18, right: 22, bottom: 42, left: 46 };
const plotW = W - PAD.left - PAD.right;
const plotH = H - PAD.top - PAD.bottom;

const yPos = (v: number, ymax: number) => PAD.top + plotH * (1 - v / ymax);
// Line points sit at the band edges (first at left, last at right).
const xLine = (i: number, n: number) =>
  PAD.left + (n === 1 ? plotW / 2 : (plotW * i) / (n - 1));
// Bars sit centred in equal bands.
const xBandCenter = (i: number, n: number) => PAD.left + (plotW * (i + 0.5)) / n;

function Frame({
  categories,
  yTicks,
  ymax,
  children,
}: {
  categories: string[];
  yTicks: number[];
  ymax: number;
  children: React.ReactNode;
}) {
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
    >
      {/* gridlines + y labels */}
      <g className="text-fd-border" stroke="currentColor" strokeOpacity={0.5}>
        {yTicks.map((t) => (
          <line
            key={t}
            x1={PAD.left}
            x2={W - PAD.right}
            y1={yPos(t, ymax)}
            y2={yPos(t, ymax)}
            strokeDasharray="3 3"
          />
        ))}
      </g>
      <g
        className="text-fd-muted-foreground"
        fill="currentColor"
        fontSize={11}
        textAnchor="end"
      >
        {yTicks.map((t) => (
          <text key={t} x={PAD.left - 8} y={yPos(t, ymax) + 4}>
            {t}%
          </text>
        ))}
      </g>
      {/* x labels */}
      <g
        className="text-fd-muted-foreground"
        fill="currentColor"
        fontSize={11}
        textAnchor="middle"
      >
        {categories.map((c, i) => (
          <text
            key={c}
            x={
              categories.length > 1 && plotW > 0
                ? xBandCenter(i, categories.length)
                : xLine(i, categories.length)
            }
            y={H - PAD.bottom + 22}
          >
            {c}
          </text>
        ))}
      </g>
      {children}
    </svg>
  );
}

function LineSeries({
  data,
  ymax,
  color,
  dashed,
}: {
  data: number[];
  ymax: number;
  color: string;
  dashed?: boolean;
}) {
  const pts = data.map((v, i) => [xLine(i, data.length), yPos(v, ymax)]);
  const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0]},${p[1]}`).join(' ');
  return (
    <g>
      <path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth={2.75}
        strokeDasharray={dashed ? '7 5' : undefined}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {pts.map((p, i) => (
        <circle key={i} cx={p[0]} cy={p[1]} r={4.5} fill={color} />
      ))}
    </g>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
  footer,
  legend,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
  footer?: string;
  legend?: { label: string; color: string; dashed?: boolean }[];
}) {
  return (
    <div className="w-full rounded-xl border bg-fd-card p-5 sm:p-6">
      <div className="mb-4">
        <h4 className="text-base font-semibold sm:text-lg">{title}</h4>
        <p className="mt-1 text-xs text-fd-muted-foreground sm:text-sm">
          {subtitle}
        </p>
      </div>
      {children}
      {legend && (
        <div className="mt-4 flex flex-wrap justify-center gap-x-5 gap-y-2">
          {legend.map((l) => (
            <span
              key={l.label}
              className="flex items-center gap-2 text-xs text-fd-muted-foreground"
            >
              <svg width={22} height={8} aria-hidden>
                <line
                  x1={0}
                  y1={4}
                  x2={22}
                  y2={4}
                  stroke={l.color}
                  strokeWidth={3}
                  strokeDasharray={l.dashed ? '5 3' : undefined}
                />
              </svg>
              {l.label}
            </span>
          ))}
        </div>
      )}
      {footer && (
        <p className="mt-3 text-center text-xs text-fd-muted-foreground">
          {footer}
        </p>
      )}
    </div>
  );
}

function BarChart({
  data,
  ymax,
  yTicks,
}: {
  data: { name: string; score: number; fill: string }[];
  ymax: number;
  yTicks: number[];
}) {
  const bandW = plotW / data.length;
  const barW = Math.min(84, bandW * 0.5);
  return (
    <Frame categories={data.map((d) => d.name)} yTicks={yTicks} ymax={ymax}>
      {data.map((d, i) => {
        const cx = xBandCenter(i, data.length);
        const top = yPos(d.score, ymax);
        const base = yPos(0, ymax);
        return (
          <g key={d.name}>
            <rect
              x={cx - barW / 2}
              y={top}
              width={barW}
              height={base - top}
              rx={6}
              fill={d.fill}
            />
            <text
              x={cx}
              y={top - 7}
              textAnchor="middle"
              fontSize={12}
              fontWeight={600}
              fill={d.fill}
            >
              {d.score}%
            </text>
          </g>
        );
      })}
    </Frame>
  );
}

const STAGES = ['After stage 0', 'After stage 1', 'After stage 2'];

export function CatastrophicForgettingChart() {
  return (
    <ChartCard
      title="Catastrophic forgetting with standard fine-tuning"
      subtitle="Accuracy on stage-0 queries (Airtable / GitHub / Notion) after each training stage · SFT baseline"
      footer="61.6% → 5.7% on the same held-out Airtable queries — while the model learns each new toolkit."
    >
      <Frame categories={STAGES} yTicks={[0, 20, 40, 60]} ymax={70}>
        <LineSeries data={[61.6, 38.4, 5.7]} ymax={70} color={FORGET} />
      </Frame>
    </ChartCard>
  );
}

export function RetentionAfterStage2Chart() {
  return (
    <ChartCard
      title="Stage-0 accuracy after learning stages 1 and 2"
      subtitle="val_stage0 after stage 2 · zero-shot, except SDFT* (few-shot eval matching training)"
    >
      <BarChart
        ymax={70}
        yTicks={[0, 20, 40, 60]}
        data={[
          { name: 'SFT', score: 5.7, fill: SFT },
          { name: 'Plain SDFT*', score: 15.7, fill: PLAIN },
          { name: 'Hybrid α=0.4', score: 31.4, fill: HYBRID },
          { name: 'Multi-teacher', score: 61.6, fill: MULTI },
        ]}
      />
    </ChartCard>
  );
}

export function ForgettingProgressionChart() {
  return (
    <ChartCard
      title="Does stage-0 knowledge survive new toolkits?"
      subtitle="val_stage0 after each stage · SDFT* uses few-shot eval; others zero-shot"
      legend={[
        { label: 'SFT', color: SFT },
        { label: 'Plain SDFT*', color: PLAIN, dashed: true },
        { label: 'Hybrid SDFT+SFT', color: HYBRID },
        { label: 'Multi-teacher hybrid', color: MULTI },
      ]}
    >
      <Frame categories={STAGES} yTicks={[0, 20, 40, 60]} ymax={70}>
        <LineSeries data={[61.6, 38.4, 5.7]} ymax={70} color={SFT} />
        <LineSeries data={[37.7, 20.1, 15.7]} ymax={70} color={PLAIN} dashed />
        <LineSeries data={[63.5, 44.7, 31.4]} ymax={70} color={HYBRID} />
        <LineSeries data={[58.5, 58.5, 61.6]} ymax={70} color={MULTI} />
      </Frame>
    </ChartCard>
  );
}

export function FullSplitRetentionChart() {
  return (
    <ChartCard
      title="Harder held-out split (full_stage0) after all 3 stages"
      subtitle="~1,800 stage-0 queries · zero-shot, except SDFT* (few-shot)"
    >
      <BarChart
        ymax={55}
        yTicks={[0, 20, 40]}
        data={[
          { name: 'SFT', score: 0.5, fill: SFT },
          { name: 'Plain SDFT*', score: 7.5, fill: PLAIN },
          { name: 'Hybrid', score: 28.5, fill: HYBRID },
          { name: 'Multi-teacher', score: 49.5, fill: MULTI },
        ]}
      />
    </ChartCard>
  );
}
