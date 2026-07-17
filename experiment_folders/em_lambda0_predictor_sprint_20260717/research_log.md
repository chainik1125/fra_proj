# Research log

Append only after the sprint starts. Each entry must include UTC/KST time, elapsed and
remaining wall time, spend/commitment, live remote resources, decision, evidence, and
next action.

## Preflight

- Isolated branch: `dmitry/em/codex-lambda0-remote` from `ef17268cc8ddd355bf2b35a36fdc5eb39f68edd3`.
- Primary compute: Modal. RunPod fallback is mediated by a Modal CPU controller because
  the existing RunPod backend relies on raw SSH.
- Provider credentials were escrowed locally as the named Modal secrets
  `em-sprint-judges` and `em-sprint-runpod`. Codex Cloud needs only a fresh Modal
  token during setup; values must never be copied into this log.
- The source checkout contained an active, dirty Claude sprint. It is prior work and
  must not be modified, staged, or used as an unlabelled prospective result.
