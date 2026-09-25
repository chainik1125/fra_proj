# Compute request board

The local poller reads only commits pushed to `dmitry/cadenza-llamascope-overnight-20260921`.
It checks this file every two minutes and launches a request once when its status is `READY` and
the resource and source checks pass. Keep requests as separate sections and give every attempt a
new unique ID.

Prose-only requests such as the original REQ-1 are not parsed or launched automatically. Convert a
request to the JSON format below, and push it with `status` set to `READY` only when its code is
also present in that commit.

The active REQ-1 training run is also reported to this board every five minutes. The progress
publisher stops after it posts the run's final state.

Put each request's Python code under `experiments/message-board/<id>/`. The poller runs that
directory's code from the exact pushed commit it observed. It also includes the shared `fra/`
package and `requirements.txt` in the remote job snapshot.

Use this format in `indranil-requests.md`:

````markdown
## request: indranil-20260924-01

```json
{
  "id": "indranil-20260924-01",
  "status": "DRAFT",
  "host": "auto",
  "gpus": 1,
  "entrypoint": "experiments/message-board/indranil-20260924-01/run.py",
  "args": ["--seed", "0"],
  "timeout_minutes": 720
}
```
````

Change `status` to `READY` only after the code and request are pushed. `host` must be `auto`,
`simplex1`, `simplex2`, or `simplex3`; `gpus` must be 1 or 2; and `timeout_minutes` must be
between 1 and 4320. A two-GPU request uses both devices on one host. The bridge chooses devices
that look idle, exposes only those through `CUDA_VISIBLE_DEVICES`, and keeps at most two GPUs
reserved across all bridge jobs. It stops and waits if any Simplex host cannot be checked.

The job runs with `/data/users/dmitry/simplex-research/.venv/bin/python`; dependencies are not
installed automatically. The job receives `JOB_ID`, `JOB_SOURCE_COMMIT`, and `JOB_OUTPUT_DIR`.
Its output and logs stay on the selected host at
`/data/users/dmitry/fra-compute-bridge/jobs/<id>-<commit-prefix>/`.

The request is launched at most once per ID, including failed launches. Use a new ID to retry.
The poller does not execute shell snippets or commands from this Markdown file.
