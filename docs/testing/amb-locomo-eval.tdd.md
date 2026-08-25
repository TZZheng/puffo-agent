# AMB LoCoMo provider evidence

## Source plan

`docs/amb-locomo-eval-plan.md` (pre-existing, untracked; not modified by this work).

## Guarantees

| Guarantee | Test | Result |
| --- | --- | --- |
| BM25 ranks the relevant stored note and returns the complete stored body | `test_bm25_retrieval_ranks_relevant_note_and_returns_full_body` | PASS |
| Retrieval never crosses the requested LoCoMo conversation/user namespace | `test_retrieval_scopes_documents_to_the_requested_user` | PASS |
| The substring baseline is case-insensitive and respects `k` | `test_substring_mode_uses_case_insensitive_matching_and_honors_limit` | PASS |

## TDD evidence

- RED: `uv run --extra dev pytest tests/test_amb_provider.py` failed during collection with `ModuleNotFoundError: No module named 'puffo_agent.evaluation'`.
- GREEN: `uv run --extra dev pytest tests/test_amb_provider.py` — `3 passed in 0.23s`.
- Static checks: `uv run --extra dev ruff check src/puffo_agent/evaluation tests/test_amb_provider.py` and `python tools/check_python_structure.py` both passed.

## Reproducible AMB runs

Apply the accompanying AMB registry branch, install this Puffo checkout into
AMB's environment, then run all arms with the same dataset, split, query
limit, answer/judge model, and mode:

```text
uv sync
uv pip install -e <path-to-puffo-agent>
uv run amb run --dataset locomo --split locomo10 --memory bm25 --query-limit 50 --name locomo-bm25-q50
uv run amb run --dataset locomo --split locomo10 --memory puffo --query-limit 50 --name locomo-puffo-q50
uv run amb run --dataset locomo --split locomo10 --memory puffo-bm25 --query-limit 50 --name locomo-puffo-bm25-q50
```

`puffo` is the substring baseline; `puffo-bm25` is the ranked variant. Both
use an in-process `MemoryStore`, write 64 KiB note chunks, return full stored
note bodies, and set provider concurrency to one.

## Known gap

The judged LoCoMo run was not executed. On this Windows host, AMB's `uv sync`
fails because `hindsight-all` brings in `uvloop==0.22.1`, which explicitly does
not support Windows. It also requires a configured `GEMINI_API_KEY`. Run the
commands above on Linux/macOS (or an AMB dependency set that excludes
`uvloop`) to record accuracy, category breakdown, retrieval latency, and
context-token results.
