# Validation

- Scripts 56 and 57 returned exit code 0 on NVIDIA A100-SXM4-80GB.
- All three groups and four removal seeds completed; no seed was skipped.
- 336 raw rows match the five original method grids across 12 cases.
- Every numeric raw measurement is finite.
- All 23 archived source files still match the source manifest SHA-256 hashes.
- Analysis checks covered interpolation within a seed, the exact no-edit origin,
  and failure to reach a threshold.
- Runtime observer metadata is deduplicated by group/seed in analysis because
  Python line tracing revisits multiline print statements. Raw measurement rows
  are not duplicated.

Removal run: https://modal.com/apps/reichers-shai-c9-dmitry/main/ap-UPk77db9fcHGVWgXTz5MLA

Screen run: https://modal.com/apps/reichers-shai-c9-dmitry/main/ap-CgIk59oSfZq4Sk2KRps671

- PNG figure visually checked after aligning scales across groups.
- Both B1 Modal apps are stopped with zero tasks; unrelated apps were left alone.
