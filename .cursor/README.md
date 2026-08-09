# Cloud-agent environment

`environment.json` is what Cursor's cloud agents build their snapshot from.
`install.sh` runs **once per snapshot**, so the ~2 minute dependency install is
paid once rather than on every prompt.

## What an agent can do without any secrets

Everything hermetic, which is most of the repo: the full test suite passes with
no credentials, because every provider test stubs the HTTP layer and keeps the
parsing real. `uv run pytest -q` is the check to run.

## What needs secrets, and why they are not in here

Anything that talks to a GPU vendor: live pricing, availability, launching.
Those read credentials from the filesystem or environment and there is no
fallback — by design, since this code provisions machines that cost money.

| what | where it is read from |
|---|---|
| Verda | `~/.verda/config.json` — `{"client_id": "...", "client_secret": "..."}` |
| Vast.ai | `VAST_API_KEY`, or `~/.vast/config.json` — `{"api_key": "..."}` |
| PrimeIntellect | `~/.prime/config.json` |
| Nebius | `~/.nebius/credentials.json` |

Set these as **secrets in Cursor's agent settings**, never in the repo. Write
the file-based ones from a secret in `install.sh` if an agent genuinely needs
live access; most tasks do not.

Note that Vast's *search* endpoint needs no key, so availability and pricing
work unauthenticated there — only renting needs one.

## State the SDK writes

`~/.evsys/reliability.jsonl` (append-only launch/preemption ledger) and
`~/.evsys/queue.jsonl` (the job queue). Both are per-machine, so a cloud agent
starts with empty ones and will not see history from your laptop.

`~/.sky/catalogs/` holds SkyPilot's price CSVs. `compute.catalog.refresh()`
rewrites them from live APIs; on a fresh agent the directory is absent until
SkyPilot populates it, and refresh() no-ops rather than failing.
