import Link from 'next/link';
import {
  CatastrophicForgettingChart,
  ForgettingProgressionChart,
  FullSplitRetentionChart,
  RetentionAfterStage2Chart,
} from '@/components/continual-sdft-charts';

export const metadata = {
  title: 'Teaching an agent new tools without forgetting the old ones',
  description:
    'When a tool search model learns new app integrations, how much does it forget the old ones? We measured that tradeoff across three training stages — and three changes to the training recipe let it accumulate new knowledge without erasing the first.',
};

export default function ResearchForgettingPage() {
  return (
    <article className="mx-auto w-full max-w-[760px] px-4 py-16 sm:px-6">
      <Link
        href="/"
        className="mb-8 inline-flex items-center gap-2 text-sm text-fd-muted-foreground transition-colors hover:text-fd-foreground"
      >
        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to home
      </Link>

      <header className="mb-10">
        <div className="mb-4 flex flex-wrap items-center gap-2 text-xs font-medium text-fd-muted-foreground">
          <span className="rounded-full bg-fd-primary/10 px-2.5 py-1 text-fd-primary">
            Case study
          </span>
          <span className="opacity-60">•</span>
          <span>Continual learning</span>
          <span className="opacity-60">•</span>
          <span>Avdhoot Golekar</span>
          <span className="opacity-60">•</span>
          <span>14 min read</span>
        </div>
        <h1 className="mb-5 text-[34px] font-semibold leading-[1.1] tracking-tight sm:text-[44px] lg:text-[52px]">
          Teaching an agent new tools without forgetting the old ones
        </h1>
        <p className="text-base leading-relaxed text-fd-muted-foreground sm:text-lg">
          Every time you fine-tune an agent on a new set of tools, it picks up the new
          capabilities but often at the cost of the old ones. We ran a three stage benchmark
          to measure exactly how much earlier knowledge gets lost while new knowledge
          accumulates, then tried four training recipes to see if that tradeoff is inevitable.
        </p>
      </header>

      <div className="mb-12 rounded-r-lg border-l-4 border-fd-primary/60 bg-fd-primary/5 px-5 py-5 sm:px-6 sm:py-6">
        <div className="mb-3 flex items-center gap-2">
          <span className="text-sm font-semibold uppercase tracking-wider text-fd-primary">
            TL;DR
          </span>
        </div>
        <p className="text-[15px] leading-relaxed sm:text-base">
          Fine tune a tool search model on three batches of app integrations (Airtable → Gmail →
          Slack) and it forgets the first batch. Standard SFT drops stage-0 accuracy from{' '}
          <strong className="font-semibold">62% to 6%</strong>. Soft distillation
          (SDFT) forgets less but peaks lower. Three changes: warm-start teacher, a hybrid
          hard+soft loss, and frozen multi-teacher ensembles bring stage-0 accuracy back to{' '}
          <strong className="font-semibold">62%</strong> with no extra trainable parameters.
        </p>
      </div>

      {/* 1. Premise */}
      <section className="mb-14">
        <h2 className="mb-4 text-2xl font-semibold tracking-tight sm:text-3xl">
          The problem we wanted to solve — Catastrophic Forgetting
        </h2>
        <p className="mb-4 leading-relaxed text-fd-muted-foreground">
          Agents in production keep getting new integrations. This week you ship Airtable. Next
          month, Gmail. Then Slack. Each time, you fine tune the model on the new tools and deploy.
        </p>
        <p className="mb-4 leading-relaxed text-fd-muted-foreground">
          After you train on Gmail, does the model still know how to pick an Airtable tool? Or did
          learning the new stuff overwrite the old stuff?
        </p>
        <p className="mb-5 leading-relaxed text-fd-muted-foreground">
          Here is a concrete example of what the model has to do. A user says:
        </p>
        <div className="mb-5 rounded-xl border bg-fd-card px-5 py-4">
          <p className="font-mono text-sm">
            &ldquo;Pull up a specific airtable record by id&rdquo;
          </p>
        </div>
        <p className="mb-3 leading-relaxed text-fd-muted-foreground">
          The model should answer with the right API slug, wrapped in tags:
        </p>
        <div className="mb-5 rounded-xl border bg-fd-card px-5 py-4 font-mono text-sm">
          &lt;answer&gt;AIRTABLE_GET_RECORD&lt;/answer&gt;
        </div>
        <p className="mb-4 leading-relaxed text-fd-muted-foreground">
          That part works fine after stage 0 training. The model hits about 62% accuracy on
          held-out Airtable queries. Then we train it on Gmail tools for another 100 steps.
          Accuracy on those same Airtable queries drops to 38%. Train on Slack next, and
          Airtable accuracy falls to 6%.
        </p>
        <p className="mb-6 leading-relaxed text-fd-muted-foreground">
          The model mostly forgot how to do Airtable queries. That is catastrophic forgetting, and
          it is the default outcome when you keep fine-tuning the same weights on new data.
        </p>

        <CatastrophicForgettingChart />
      </section>

      {/* 2. Setup */}
      <section className="mb-14">
        <h2 className="mb-4 text-2xl font-semibold tracking-tight sm:text-3xl">
          How we set up the experiment
        </h2>
        <p className="mb-4 leading-relaxed text-fd-muted-foreground">
          We built a benchmark called composio-bench. The task is tool search: read a natural
          language query, output the correct Composio API slug. We used Qwen3.5-4B with a small
          LoRA adapter (rank 8). Same base model throughout, only the adapter weights change
          between stages.
        </p>
        <p className="mb-6 leading-relaxed text-fd-muted-foreground">
          Training happens in three stages. Each stage introduces a new set of app integrations.
          The model trains for 100 steps on that stage&apos;s data, we save a checkpoint, then
          move on. We never retrain on old data unless a recipe explicitly says to.
        </p>

        <div className="mb-6 overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b">
                <th className="px-3 py-3 text-left font-semibold">Stage</th>
                <th className="px-3 py-3 text-left font-semibold">New toolkits added</th>
                <th className="px-3 py-3 text-left font-semibold">Example query</th>
              </tr>
            </thead>
            <tbody className="text-fd-muted-foreground">
              <tr className="border-b">
                <td className="px-3 py-3 align-top font-medium text-fd-foreground">Stage 0</td>
                <td className="px-3 py-3 align-top">Airtable, GitHub, Notion</td>
                <td className="px-3 py-3 align-top font-mono text-xs">
                  &ldquo;Make a fresh table in an airtable base&rdquo;
                  <span className="mt-1 block font-sans">→ AIRTABLE_CREATE_TABLE</span>
                </td>
              </tr>
              <tr className="border-b">
                <td className="px-3 py-3 align-top font-medium text-fd-foreground">Stage 1</td>
                <td className="px-3 py-3 align-top">Gmail, HubSpot, Instagram, Linear, Supabase, Twitter</td>
                <td className="px-3 py-3 align-top font-mono text-xs">
                  &ldquo;Send an email with gmail&rdquo;
                  <span className="mt-1 block font-sans">→ GMAIL_SEND_EMAIL</span>
                </td>
              </tr>
              <tr>
                <td className="px-3 py-3 align-top font-medium text-fd-foreground">Stage 2</td>
                <td className="px-3 py-3 align-top">Google Calendar, Drive, Sheets, Outlook, Slack, Telegram</td>
                <td className="px-3 py-3 align-top font-mono text-xs">
                  &ldquo;Post a message to a slack channel&rdquo;
                  <span className="mt-1 block font-sans">→ SLACK_SEND_MESSAGE</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <h3 className="mb-3 text-xl font-semibold">What is validation accuracy?</h3>
        <p className="mb-4 leading-relaxed text-fd-muted-foreground">
          After each stage, we test the model on queries it did <em>not</em> see during training.
          Each stage has its own held-out validation set (about 10% of that stage&apos;s queries,
          split by toolkit).
        </p>
        <p className="mb-4 leading-relaxed text-fd-muted-foreground">
          When we write <code className="text-fd-primary">val_stage0 = 61.6%</code>, we mean:
        </p>
        <ul className="mb-4 list-disc space-y-2 pl-6 text-fd-muted-foreground">
          <li>
            Take the model <em>as it exists right now</em> (after however many stages of training
            have run).
          </li>
          <li>Run it on the stage-0 validation queries (Airtable / GitHub / Notion only).</li>
          <li>
            Count how often the output slug exactly matches the correct answer. That fraction is
            validation accuracy.
          </li>
        </ul>
        <p className="mb-4 leading-relaxed text-fd-muted-foreground">
          So if you see <code className="text-fd-primary">val_stage0</code> drop from 62% after
          stage 0 to 6% after stage 2, that means: the model finished learning Slack tools, and
          when you ask it an Airtable question it used to handle well, it gets it wrong 94% of the
          time.
        </p>
        <p className="leading-relaxed text-fd-muted-foreground">
          We also report <code className="text-fd-primary">full_stage0</code>,{' '}
          <code className="text-fd-primary">full_stage1</code>, and{' '}
          <code className="text-fd-primary">full_stage2</code> — larger held-out splits (~200
          queries each) drawn from the full Composio eval suite, stratified by toolkit. Same pass@1
          metric, harder distribution, more stable than the small val sets. When we say a result
          &ldquo;held up on the full benchmark,&rdquo; we mean these numbers.
        </p>
      </section>

      {/* 3. Recipes */}
      <section className="mb-14">
        <h2 className="mb-4 text-2xl font-semibold tracking-tight sm:text-3xl">
          Four training recipes
        </h2>
        <p className="mb-6 leading-relaxed text-fd-muted-foreground">
          We ran the same 3 stage benchmark four ways. Every time we change the way the model
          learns, note that the model and the dataset remains the same.
        </p>

        <div className="space-y-8">
          <div>
            <h3 className="mb-2 text-lg font-semibold">Recipe 1: Standard fine-tuning (SFT)</h3>
            <p className="leading-relaxed text-fd-muted-foreground">
              The baseline. Show the model a query and the correct answer slug. Train it to predict
              that slug directly. This is what most people do. It learns the current stage fast and
              forgets previous stages just as fast, that is the 62% → 6% drop we opened with.
            </p>
          </div>
          <div>
            <h3 className="mb-2 text-lg font-semibold">Recipe 2: Self-distillation (SDFT)</h3>
            <p className="mb-3 leading-relaxed text-fd-muted-foreground">
              Instead of hard labels, the student model first writes its own answer. A teacher model
              then scores that answer token-by-token (with the correct answer shown as an in-context
              example). The student learns to match the teacher&apos;s softer probability
              distribution rather than memorizing one exact slug. This way we preserve hinton&apos;s
              dark knowledge.
            </p>
            <p className="leading-relaxed text-fd-muted-foreground">
              SDFT did help with forgetting. After all three stages, stage-0 accuracy was 15.7% vs.
              SFT&apos;s 5.7% — nearly 3× better on the same metric. Relative to each method&apos;s
              own peak, SDFT kept about 42% of stage-0 knowledge (37.7% → 15.7%) while SFT kept only
              9% (61.6% → 5.7%). The catch: SDFT never peaked as high as SFT (37.7% vs. 61.6% after
              stage 0), because training used a few-shot format scaffold. Zero-shot eval read as 0%
              — an eval mismatch, not proof the model learned nothing. Re-evaluating with the same
              few-shot prompt used in training showed real capability and milder forgetting. Still
              not good enough to ship zero-shot, which is why we kept iterating.
            </p>
          </div>
          <div>
            <h3 className="mb-2 text-lg font-semibold">Recipe 3: Hybrid SDFT + SFT</h3>
            <p className="leading-relaxed text-fd-muted-foreground">
              Mix two losses: 40% standard cross-entropy on the correct answer, 60% distillation
              from the teacher. The hard labels keep the output format stable. The soft distillation
              keeps the weight updates gentle. Same trainable parameters as before.
            </p>
          </div>
          <div>
            <h3 className="mb-2 text-lg font-semibold">Recipe 4: Multi-teacher hybrid</h3>
            <p className="leading-relaxed text-fd-muted-foreground">
              Same hybrid loss, plus: after each stage, freeze that stage&apos;s adapter as a
              read-only teacher. On new data, average predictions from all past teachers plus the
              current one. On old data (25% of each batch), replay queries from prior stages and
              distill from the specialist teacher for that stage only.
            </p>
          </div>
        </div>
      </section>

      {/* 4. Results */}
      <section className="mb-14 space-y-6">
        <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">Results</h2>
        <p className="leading-relaxed text-fd-muted-foreground">
          The number we care about most is <code className="text-fd-primary">val_stage0</code> after
          all three stages are done: after learning everything, how good is the model at the{' '}
          <em>first</em> toolkit set? SFT, hybrid, and multi-teacher use zero-shot eval (same prompt
          as production). Plain SDFT trained with a few-shot scaffold, so we report it with a
          matching few-shot eval.
        </p>

        <RetentionAfterStage2Chart />

        <p className="leading-relaxed text-fd-muted-foreground">
          SFT ends at 5.7%. Plain SDFT at 15.7% (few-shot eval), lower peak, but roughly 3× the
          retained stage-0 accuracy and a much gentler slide from peak to final. Hybrid gets to
          31.4%. Multi-teacher gets to 61.6%.
        </p>

        <ForgettingProgressionChart />

        <p className="leading-relaxed text-fd-muted-foreground">
          The dashed SDFT line shows the same pattern: accuracy drops as new stages land, but the
          slope is flatter than SFT. SDFT was pointed in the right direction on forgetting; hybrid
          and multi-teacher built on that and fixed the zero-shot deploy gap.
        </p>

        <FullSplitRetentionChart />

        <h3 className="mt-8 text-xl font-semibold">Full benchmark after stage 2</h3>
        <p className="leading-relaxed text-fd-muted-foreground">
          Complete pass@1 on every val and full split, after all three training stages. This is the
          cleanest side-by-side: rows are recipes, columns are which toolkit era each eval probes.
        </p>

        <div className="overflow-x-auto rounded-xl border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-fd-muted/30">
                <th className="px-3 py-3 text-left font-medium">Recipe</th>
                <th className="px-3 py-3 text-right font-medium">val_s0</th>
                <th className="px-3 py-3 text-right font-medium">val_s1</th>
                <th className="px-3 py-3 text-right font-medium">val_s2</th>
                <th className="px-3 py-3 text-right font-medium">full_s0</th>
                <th className="px-3 py-3 text-right font-medium">full_s1</th>
                <th className="px-3 py-3 text-right font-medium">full_s2</th>
              </tr>
            </thead>
            <tbody className="text-fd-muted-foreground">
              <tr className="border-b">
                <td className="px-3 py-3">SFT</td>
                <td className="px-3 py-3 text-right">5.7%</td>
                <td className="px-3 py-3 text-right">21.2%</td>
                <td className="px-3 py-3 text-right">40.2%</td>
                <td className="px-3 py-3 text-right">0.5%</td>
                <td className="px-3 py-3 text-right">18.5%</td>
                <td className="px-3 py-3 text-right">43.5%</td>
              </tr>
              <tr className="border-b">
                <td className="px-3 py-3">Plain SDFT*</td>
                <td className="px-3 py-3 text-right">15.7%</td>
                <td className="px-3 py-3 text-right">23.0%</td>
                <td className="px-3 py-3 text-right">23.2%</td>
                <td className="px-3 py-3 text-right">7.5%</td>
                <td className="px-3 py-3 text-right">27.5%</td>
                <td className="px-3 py-3 text-right">25.0%</td>
              </tr>
              <tr className="border-b">
                <td className="px-3 py-3">Hybrid (α=0.4)</td>
                <td className="px-3 py-3 text-right">31.4%</td>
                <td className="px-3 py-3 text-right">28.5%</td>
                <td className="px-3 py-3 text-right">39.6%</td>
                <td className="px-3 py-3 text-right">28.5%</td>
                <td className="px-3 py-3 text-right">36.0%</td>
                <td className="px-3 py-3 text-right">48.0%</td>
              </tr>
              <tr>
                <td className="px-3 py-3 font-medium text-fd-foreground">Multi-teacher hybrid</td>
                <td className="px-3 py-3 text-right font-medium text-emerald-600 dark:text-emerald-400">61.6%</td>
                <td className="px-3 py-3 text-right">35.2%</td>
                <td className="px-3 py-3 text-right">36.6%</td>
                <td className="px-3 py-3 text-right font-medium text-emerald-600 dark:text-emerald-400">49.5%</td>
                <td className="px-3 py-3 text-right">43.5%</td>
                <td className="px-3 py-3 text-right">49.0%</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p className="text-sm text-fd-muted-foreground">
          * Plain SDFT: few-shot eval (matches training prompt). Zero-shot eval was 0% across all
          stages — a format mismatch, not zero capability. SFT, hybrid, and multi-teacher: zero-shot
          throughout. Per-stage numbers for all recipes are in the{' '}
          <a href="#appendix" className="text-fd-primary hover:underline">appendix</a>.
        </p>
      </section>

      {/* 5. Why */}
      <section className="mb-14 space-y-8">
        <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Why these changes worked
        </h2>
        <p className="leading-relaxed text-fd-muted-foreground">
          Multi-teacher did not add parameters. It is still one LoRA adapter being trained. The
          gains came from three fixes we made while iterating on SDFT. Each one addresses a specific
          failure mode we hit.
        </p>

        <div>
          <h3 className="mb-3 text-xl font-semibold">
            Fix 1: The teacher has to know what the student already knows
          </h3>
          <p className="mb-4 leading-relaxed text-fd-muted-foreground">
            Our first SDFT runs used the raw base model as the teacher. That sounds reasonable — the
            base model is the starting point, so distill from it. But at stage 1 the student already
            learned Airtable. The base model did not. Every distillation step was pulling the student
            back toward generic pretraining behavior.
          </p>
          <p className="leading-relaxed text-fd-muted-foreground">
            The fix: at the start of each stage, snapshot the student&apos;s current weights and use{' '}
            <em>that</em> as the teacher. Stage 0 teacher = base model. Stage 1 teacher = stage-0
            checkpoint. The teacher always represents what the model knew going in, plus an
            in-context demo of the right answer. Distillation now means &ldquo;stay close to what you
            already know, adjusted by this example&rdquo; instead of &ldquo;forget everything and
            start over.&rdquo;
          </p>
        </div>

        <div>
          <h3 className="mb-3 text-xl font-semibold">
            Fix 2: Hard labels for format, soft targets for retention
          </h3>
          <p className="mb-4 leading-relaxed text-fd-muted-foreground">
            Pure distillation is good at not overwriting weights. It is bad at locking in output
            format when training includes a few-shot example the model never sees at deploy time. Our
            first SDFT runs looked like total failure in zero-shot eval (0% everywhere). Re-running
            eval with the training-matched few-shot prompt showed 15.7% stage-0 accuracy after stage
            2, the model had learned, we were just measuring it wrong.
          </p>
          <p className="mb-4 leading-relaxed text-fd-muted-foreground">
            SFT has the opposite problem: hard labels lock format, but the updates are aggressive
            enough to wipe old knowledge. So we split the loss:
          </p>
          <div className="mb-4 rounded-xl border bg-fd-card px-5 py-4 font-mono text-sm">
            loss = 0.4 × predict the exact slug + 0.6 × match the teacher&apos;s token distribution
          </div>
          <p className="leading-relaxed text-fd-muted-foreground">
            The 40% SFT anchor trains on zero-shot prompts (no few-shot scaffold), so training
            matches deployment. The 60% distillation keeps the update small enough that prior stages
            do not get bulldozed.
          </p>
        </div>

        <div>
          <h3 className="mb-3 text-xl font-semibold">Fix 3: Keep the specialists around</h3>
          <p className="mb-4 leading-relaxed text-fd-muted-foreground">
            Hybrid fixed a lot, but it still had one teacher that gets replaced each stage. By stage
            2 that teacher is a generalist, decent at everything, sharp on nothing. Stage-0 knowledge
            is in there somewhere, but diluted.
          </p>
          <p className="mb-4 leading-relaxed text-fd-muted-foreground">
            Multi-teacher keeps a frozen copy of each stage&apos;s adapter (~30 MB each, not a full
            4B model). Two rules for how to use them:
          </p>
          <ul className="mb-4 list-disc space-y-2 pl-6 text-fd-muted-foreground">
            <li>
              <strong className="text-fd-foreground">New queries (75% of batch):</strong> ask every
              frozen teacher plus the current one what they think. Average their token-level
              predictions. The student learns from the consensus of all its past selves.
            </li>
            <li>
              <strong className="text-fd-foreground">Old queries (25% of batch):</strong> replay
              stage-0 or stage-1 examples and distill from the specialist for that stage only.
              Airtable rows go to the stage-0 teacher. Gmail rows go to the stage-1 teacher.
            </li>
          </ul>
          <p className="leading-relaxed text-fd-muted-foreground">
            You are not storing three full models. You are storing three small snapshots, each frozen
            at the point where the model was best at that stage&apos;s toolkits. Replay keeps those
            snapshots relevant. The ensemble keeps new learning from drifting too far from all of
            them at once.
          </p>
        </div>
      </section>

      <section className="mb-14">
        <h2 className="mb-4 text-2xl font-semibold tracking-tight sm:text-3xl">
          What this means in practice
        </h2>
        <p className="mb-4 leading-relaxed text-fd-muted-foreground">
          If you are shipping new agent capabilities on a schedule — new integrations, new tool
          sets, new domains — the default fine-tuning loop will eat its own tail. You do not
          necessarily need more data from old stages or a bigger model. You need to think about who
          teaches the model, on what data, and whether old teachers stick around.
        </p>
        <p className="mb-4 leading-relaxed text-fd-muted-foreground">
          Multi-teacher adds about 10% wall-clock time (extra teacher queries per batch) and ~60 MB
          of frozen adapters at stage 2. No extra trainable parameters.
        </p>
        <p className="leading-relaxed text-fd-muted-foreground">
          We are running follow-ups — multi-teacher at 9B, a second seed, and wiring these knobs into
          our autoresearch loop so a coding agent can search over teacher configs automatically. More
          in our{' '}
          <a
            href="https://evolvingsystems.ai/research/continual-learning-is-a-system"
            className="text-fd-primary hover:underline"
          >
            continual learning system post
          </a>
          .
        </p>
      </section>

      {/* Appendix */}
      <section id="appendix" className="mb-14 scroll-mt-28 border-t pt-10">
        <h2 className="mb-2 text-2xl font-semibold tracking-tight sm:text-3xl">
          Appendix: complete benchmark log
        </h2>
        <p className="mb-8 text-sm text-fd-muted-foreground">
          Pass@1 on every val and full split, after each training stage. Read across a row to see
          what the model knew at that checkpoint; read down a column to see forgetting on one toolkit
          era as later stages land.
        </p>

        <div className="space-y-6">
          {[
            {
              name: 'SFT (zero-shot)',
              rows: [
                { stage: 'After stage 0', val: ['61.6', '17.6', '7.9'], full: ['42.0', '30.5', '17.0'] },
                { stage: 'After stage 1', val: ['38.4', '36.4', '4.3'], full: ['32.0', '48.5', '15.0'] },
                { stage: 'After stage 2', val: ['5.7', '21.2', '40.2'], full: ['0.5', '18.5', '43.5'] },
              ],
            },
            {
              name: 'Plain SDFT* (few-shot eval)',
              rows: [
                { stage: 'After stage 0', val: ['37.7', '14.5', '8.5'], full: ['24.5', '23.0', '13.0'] },
                { stage: 'After stage 1', val: ['20.1', '32.7', '4.9'], full: ['15.0', '38.5', '15.5'] },
                { stage: 'After stage 2', val: ['15.7', '23.0', '23.2'], full: ['7.5', '27.5', '25.0'] },
              ],
            },
            {
              name: 'Hybrid (α=0.4, zero-shot)',
              rows: [
                { stage: 'After stage 0', val: ['63.5', '17.6', '8.5'], full: ['49.5', '25.5', '18.5'] },
                { stage: 'After stage 1', val: ['44.7', '41.8', '11.0'], full: ['40.5', '45.5', '19.0'] },
                { stage: 'After stage 2', val: ['31.4', '28.5', '39.6'], full: ['28.5', '36.0', '48.0'] },
              ],
            },
            {
              name: 'Multi-teacher hybrid (zero-shot)',
              rows: [
                { stage: 'After stage 0', val: ['58.5', '19.4', '9.1'], full: ['49.0', '30.5', '18.5'] },
                { stage: 'After stage 1', val: ['58.5', '41.8', '12.8'], full: ['46.5', '49.0', '21.5'] },
                { stage: 'After stage 2', val: ['61.6', '35.2', '36.6'], full: ['49.5', '43.5', '49.0'] },
              ],
            },
          ].map((recipe) => (
            <div key={recipe.name} className="overflow-x-auto rounded-xl border">
              <div className="border-b bg-fd-muted/20 px-4 py-3 text-sm font-medium">
                {recipe.name}
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="px-3 py-2 text-left font-medium">Checkpoint</th>
                    <th className="px-3 py-2 text-right font-medium">val_s0</th>
                    <th className="px-3 py-2 text-right font-medium">val_s1</th>
                    <th className="px-3 py-2 text-right font-medium">val_s2</th>
                    <th className="px-3 py-2 text-right font-medium">full_s0</th>
                    <th className="px-3 py-2 text-right font-medium">full_s1</th>
                    <th className="px-3 py-2 text-right font-medium">full_s2</th>
                  </tr>
                </thead>
                <tbody className="text-fd-muted-foreground">
                  {recipe.rows.map((row) => (
                    <tr key={row.stage} className="border-b last:border-0">
                      <td className="px-3 py-2">{row.stage}</td>
                      {row.val.map((v, i) => (
                        <td key={`val-${i}`} className="px-3 py-2 text-right">{v}%</td>
                      ))}
                      {row.full.map((v, i) => (
                        <td key={`full-${i}`} className="px-3 py-2 text-right opacity-80">{v}%</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>

        <p className="mt-6 text-sm text-fd-muted-foreground">
          * Plain SDFT evaluated with few-shot prompt (matches training). All others zero-shot.
          Qwen3.5-4B · LoRA rank 8 · 100 steps/stage.
        </p>
      </section>

      <footer className="border-t pt-8">
        <p className="text-sm text-fd-muted-foreground">
          Model: Qwen/Qwen3.5-4B · Benchmark: composio-bench continual toolkits v1 · Trained with{' '}
          <Link href="/docs" className="text-fd-primary hover:underline">
            evsys-sdk
          </Link>
          .
        </p>
      </footer>
    </article>
  );
}
