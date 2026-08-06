say(){ curl -sS -H "Title: $1" -d "$2" "$EVSYS_EVENTS_URL" >/dev/null 2>&1 || true; }
sayfile(){ local t="$1" f="$2"; [ -s "$f" ] || { say "$t" "EMPTY"; return; }
  gzip -c "$f" | base64 -w0 > /tmp/enc; local sz=$(stat -c%s /tmp/enc); local n=$(( (sz+2999)/3000 ))
  for i in $(seq 0 $((n-1))); do say "$t.$((i+1))of$n" "$(dd if=/tmp/enc bs=3000 skip=$i count=1 2>/dev/null)"; sleep 2; done; }
say "N:$EVSYS_JOB_ID" "9B agent up $(hostname); resume_step=$EVSYS_RESUME_STEP; vol=$EVSYS_VOLUME"
( while true; do sleep 120; say "hb:$EVSYS_JOB_ID" "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';') run:$(tail -c 200 /root/evsys_run.log 2>/dev/null | tr '\n' ' ') srv:$(tail -c 300 /root/srv.log 2>/dev/null | tr '\n' ' ')"; done ) &
# CUDA toolkit on PATH: TileLang JIT for the GDN kernels needs nvcc 12.x
# (the apt nvcc 11.5 lacks <cuda/atomic>); the image has 12.8 off-PATH.
export PATH=/usr/local/cuda/bin:/root/.local/bin:$PATH CUDA_HOME=/usr/local/cuda HF_HUB_ENABLE_HF_TRANSFER=1
# GDN kernel backend: TileLang is the intended backend on Hopper (skyrl
# train/utils/utils.py). Persist its JIT cache on the volume so a preempted
# node's replacement skips recompilation.
export FLA_TILELANG=1 TILELANG_CACHE_DIR=/data/tilelang_cache
mkdir -p /data/tilelang_cache && ln -sfn /data/tilelang_cache /root/.tilelang
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
for i in 1 2 3 4; do [ -d /root/skyrl/.git ] && break; rm -rf /root/skyrl; git clone --depth 1 https://github.com/NovaSky-AI/SkyRL /root/skyrl && break; sleep $((i*5)); done
cd /root/skyrl || { say "FAIL:$EVSYS_JOB_ID" clone; exit 1; }
S=$(date +%s); uv sync --extra tinker --extra megatron > /root/sync.log 2>&1 \
  || { say "FAIL:$EVSYS_JOB_ID" "sync: $(tail -c 800 /root/sync.log | tr '\n' ' ')"; exit 1; }
say "N:$EVSYS_JOB_ID" "skyrl synced $(( $(date +%s)-S ))s"
mkdir -p /data/skyrl_ckpts /data/run
ln -sfn /data/skyrl_ckpts /tmp/skyrl_checkpoints
export SKYRL_DATABASE_URL="sqlite:////data/tinker.db"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The Qwen3.5 recipe (root-caused from skyrl source, commit 42f3d44):
#  * language_model_only trio -> native GPTModel+GDN path; fixes BOTH the
#    TP=1 init 400 (VL double-pack ValueError) and the fused-lm-head
#    'output_processor' crash (Qwen3VLModel didn't accept the hook).
#  * fused_lm_head_logprob works on this path (GPU CI runs it for 0.8B).
setsid nohup uv run --extra tinker --extra megatron python -m skyrl.tinker.api \
  --host 0.0.0.0 --port 8000 --base-model Qwen/Qwen3.5-9B \
  --backend megatron --backend-config '{
    "trainer.placement.policy_num_gpus_per_node": 1,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.colocate_all": false,
    "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
    "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
    "trainer.policy.megatron_config.lora_config.merge_lora": false,
    "trainer.policy.language_model_only": true,
    "trainer.ref.language_model_only": true,
    "generator.inference_engine.language_model_only": true,
    "trainer.logprobs_chunk_size": 1024,
    "trainer.fused_lm_head_logprob": true
  }' > /root/srv.log 2>&1 < /dev/null &
OK=""
for i in $(seq 1 240); do curl -sf --max-time 3 http://127.0.0.1:8000/api/v1/healthz >/dev/null && { OK=1; break; }; sleep 8; done
[ -n "$OK" ] || { say "FAIL:$EVSYS_JOB_ID" "server never healthy in 32min"; sayfile "SRVLOG:$EVSYS_JOB_ID" /root/srv.log; exit 1; }
say "N:$EVSYS_JOB_ID" "skyrl server healthy $(( $(date +%s)-S ))s after sync-start"
curl -sSL "__SDK_URL__" -o /root/sdk.tgz && mkdir -p /root/sdk && tar xzf /root/sdk.tgz -C /root/sdk \
  || { say "FAIL:$EVSYS_JOB_ID" "sdk fetch"; exit 1; }
uv pip install --python /root/skyrl/.venv/bin/python -q pydantic pyyaml typing-extensions requests harbor tinker-cookbook==0.4.2 > /root/pip.log 2>&1 \
  || { say "FAIL:$EVSYS_JOB_ID" "pip: $(tail -c 600 /root/pip.log | tr '\n' ' ')"; exit 1; }
cat > /root/config.yaml <<'CFGEOF'
name: e2e_router_skyrl_9b
output_dir: /data/run
run:
  name: sft9b
  data:
    source_kind: in_memory
    rows:
      - {query: "save a contact from an email I received", tool_slug: OUTLOOK_CREATE_CONTACT, toolkit: OUTLOOK, description: "Creates a new contact in Outlook."}
      - {query: "edit a slack message by timestamp", tool_slug: SLACK_UPDATES_A_SLACK_MESSAGE, toolkit: SLACK, description: "Updates a Slack message."}
      - {query: "set the topic of a Slack conversation", tool_slug: SLACK_SET_THE_TOPIC_OF_A_CONVERSATION, toolkit: SLACK, description: "Sets a channel topic."}
      - {query: "create a new issue on GitHub", tool_slug: GITHUB_CREATE_AN_ISSUE, toolkit: GITHUB, description: "Opens a new issue."}
      - {query: "send an email via Gmail", tool_slug: GMAIL_SEND_EMAIL, toolkit: GMAIL, description: "Sends an email."}
      - {query: "list my upcoming calendar events", tool_slug: GOOGLE_CALENDAR_LIST_EVENTS, toolkit: GOOGLE_CALENDAR, description: "Lists events."}
      - {query: "create a new Notion page", tool_slug: NOTION_CREATE_PAGE, toolkit: NOTION, description: "Creates a page."}
      - {query: "post a message to a Slack channel", tool_slug: SLACK_POST_MESSAGE, toolkit: SLACK, description: "Posts a message."}
    transforms:
      - kind: jsonl_to_chat
        params: {user_template: "Query: {query}", assistant_template: "<answer>{tool_slug}</answer>"}
  model:
    name: Qwen/Qwen3.5-9B
  backend:
    kind: skyrl
    params: {base_url: "http://localhost:8000", health_timeout_s: 120}
  algorithm:
    kind: sft
    params:
      max_steps: 12
      batch_size: 4
      save_every: 6
      save_sampler: false
      lora_rank: 32
CFGEOF
export TINKER_API_KEY=tml-dummy PYTHONPATH=/root/sdk/src EVSYS_PROVIDER=verda
say "N:$EVSYS_JOB_ID" "starting evsys run (9B, megatron TP=1, language_model_only)"
/root/skyrl/.venv/bin/python -m evsys_sdk.cli run /root/config.yaml > /root/evsys_run.log 2>&1
RC=$?
say "N:$EVSYS_JOB_ID" "evsys run rc=$RC"
sayfile "RUNLOG:$EVSYS_JOB_ID" /root/evsys_run.log
[ $RC -ne 0 ] && sayfile "SRVLOG:$EVSYS_JOB_ID" /root/srv.log
sleep 300
