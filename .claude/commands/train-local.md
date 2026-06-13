You are helping the user train an LLM locally using the evsys-sdk. Follow these steps carefully.

## Step 1 — Check dependencies

Run:
```bash
python -c "import trl, peft, transformers, torch, datasets; print('OK')"
```

If it fails, tell the user to install local extras first:
```bash
pip install -e ".[local]"
```
Then stop and wait for them to install before continuing.

## Step 2 — Gather training parameters

Ask the user for the following (use sensible defaults if they don't specify):

| Parameter | Default | Notes |
|---|---|---|
| Model name | `Qwen/Qwen3-0.6B` | Any HuggingFace causal LM |
| Dataset source | synthetic (10 tool-query rows) | Or a path to a JSONL file |
| Max training steps | `5` | Keep small for a first run |
| LoRA rank | `4` | Higher = more capacity, more memory |
| Output directory | `./outputs/local_run` | Where checkpoints land |

If the user provides a JSONL file path, read the first 2 rows with:
```bash
head -n 2 <path>
```
and confirm the row schema. Use the `jsonl_to_chat` transform with a `user_template`
(and optional `assistant_template`) referencing the row's fields — e.g.
`user_template="Query: {query}"`, `assistant_template="<answer>{tool_slug}</answer>"`.

## Step 3 — Generate YAML config

Write the config to `examples/configs/generated_local_sft.yaml`. Use this template,
substituting the user's values:

```yaml
name: local_sft_run
output_dir: <output_directory>
log_store:
  kind: jsonl

run:
  name: local_sft_qwen
  data:
    source_kind: in_memory          # change to jsonl + path: if user provided a file
    rows:
      - {query: "save a contact from email", tool_slug: OUTLOOK_CREATE_CONTACT, toolkit: OUTLOOK, description: "Creates a contact in Outlook."}
      - {query: "edit a slack message", tool_slug: SLACK_UPDATES_A_SLACK_MESSAGE, toolkit: SLACK, description: "Updates a Slack message."}
      - {query: "create a github issue", tool_slug: GITHUB_CREATE_AN_ISSUE, toolkit: GITHUB, description: "Opens a new issue on GitHub."}
      - {query: "send an email", tool_slug: GMAIL_SEND_EMAIL, toolkit: GMAIL, description: "Sends an email via Gmail."}
      - {query: "list calendar events", tool_slug: GOOGLE_CALENDAR_LIST_EVENTS, toolkit: GOOGLE_CALENDAR, description: "Lists upcoming calendar events."}
    transforms:
      - kind: jsonl_to_chat
        params: {user_template: "Query: {query}", assistant_template: "<answer>{tool_slug}</answer>"}
  model:
    name: <model_name>
  backend:
    kind: local
    params:
      dtype: float32
      device: cpu
  algorithm:
    kind: local_sft
    params:
      num_epochs: 1
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 1
      max_steps: <max_steps>
      max_seq_len: 256
      lora_rank: <lora_rank>
      lora_alpha: 8
      bf16: false
      fp16: false
      logging_steps: 1
      save_steps: <max_steps>
      warmup_steps: 2
  eval:
    enabled: false
```

## Step 4 — Validate

```bash
evsys validate examples/configs/generated_local_sft.yaml --deep
```

If there are errors, diagnose and fix the YAML before continuing.

## Step 5 — Run training

```bash
evsys run examples/configs/generated_local_sft.yaml -o /tmp/train_summary.json
```

Stream the output to the user. Training a 0.6B model for 5 steps on CPU takes ~3-8 minutes
on Apple Silicon. Tell the user what to expect before starting.

## Step 6 — Report results

Read the summary:
```bash
cat /tmp/train_summary.json
```

Report to the user:
- **Status**: completed / failed
- **Final loss**: from `metrics["train/loss"]` or `metrics["train/final_loss"]`
- **Checkpoint location**: from `artifacts["final_checkpoint"]`
- **Log files**: `<output_dir>/local_sft_qwen/logs/metrics.jsonl`

If status is `failed`, show the `error` field and help diagnose the issue.

## Notes for edge cases

- If the user is on a CUDA machine, suggest setting `dtype: bfloat16` and `bf16: true` in the config for faster training.
- If the model download fails (network issue), suggest `huggingface-cli login` or using a cached local path.
- If OOM occurs, suggest reducing `max_seq_len` to 128 or switching to a smaller model like `Qwen/Qwen2.5-0.5B`.
- The `jsonl_to_chat` transform builds chat rows from any row schema via `user_template`/`assistant_template`.
