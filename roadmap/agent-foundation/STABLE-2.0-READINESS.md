# Agent Foundation 2.0 Stable Readiness

This is the execution backlog for moving `puffo-agent==2.0.0a1` toward a
stable `2.0.0` release. The release contract and acceptance cases remain in
[`docs/RELEASE-CANDIDATE-2.0.0a1.md`](../../docs/RELEASE-CANDIDATE-2.0.0a1.md).

The reviewed, source-grounded inventory of every item below against the frozen
Agent source and local git evidence is in
[`roadmap/agent-foundation/STABLE-2.0-SOURCE-AUDIT.md`](STABLE-2.0-SOURCE-AUDIT.md).
Treat that audit as authoritative for classifications and evidence; this table
tracks status and evidence links.

The goal is readiness evidence and the smallest required fixes. This branch is
not a new feature integration branch.

## Missing Work

| ID | Gap | Owner / repository | Stable blocker | Completion evidence |
|---|---|---|---|---|
| A1 | The new Claude Driver launches the resolved executable directly and does not yet carry the Windows npm shim handling from Agent PR #224. | Puffo Agent | Yes | Forward-port the command normalization at the Driver boundary, keep tests limited to command construction, and complete one real Windows Claude turn. |
| S1 | Server PR #273 is still open. The metadata-only Runtime Event contract is therefore not deployed with the Agent candidate. | Puffo Server | Yes | PR merged, staging deployed, and a real Agent run confirms that only fixed-vocabulary metadata reaches the Server. |
| O1 | Production `agent_status.runtime` has not been inventoried for active Docker Codex or retired direct-runtime configurations. | Release operations | Yes | Inventory recorded and every active runtime class has an explicit migration, pin, or retirement decision. No Docker Agent silently moves to the host boundary. |
| C1 | AIM/E2B templates are not yet pinned and promoted from an immutable Agent commit for each supported harness. | Cloud Agent / AIM | Yes for cloud GA | Candidate templates pass staging, existing cloud Agent pins have a migration plan, and promotion records the exact Agent commit. |
| V1 | The final candidate SHA does not yet have one consolidated staging evidence record for upgrade, real harness, coordination, recovery, multi-target Inbox, restart, and privacy behavior. | Puffo Agent + staging | Yes | All acceptance cases in the release contract are recorded against one immutable candidate SHA. |
| R1 | Stable package metadata and publishing remain intentionally disabled. | Puffo Agent release | Yes | Only after A1, S1, O1, C1, and V1 close: set `2.0.0`, update changelog and README, tag `v2.0.0`, and run the production workflow. |

## Evidence Already Available

- `2.0.0a1` was built and clean-installed from TestPyPI from the merged
  candidate commit. Any code change after that commit requires a new candidate
  version and a fresh install check.
- Agent Foundation and its release correction are on `main` through Agent PRs
  #225 and #229.
- Agent PR #224 documents the confirmed Windows failure and the old Adapter
  fix, but remains open because the implementation must target the new Driver.

These facts reduce repeated discovery work; they do not replace final staging
evidence.

## Execution Order

1. **Agent code:** complete A1 as an isolated Driver-boundary change.
2. **Server dependency:** merge and deploy S1 before privacy acceptance.
3. **Release inventory:** complete O1 before choosing migration defaults.
4. **Candidate validation:** run V1 on the exact Agent and Server candidate
   commits. Use CLI/runtime tests for diagnosis and one browser smoke at the
   end.
5. **Cloud track:** complete C1 before calling cloud Agent support generally
   available.
6. **Stable release:** perform R1 only after every applicable gate is closed.

## Explicitly Out Of Scope

- Channel encryption and plaintext-channel policy are owned by Han's Agent PR
  #212 and must not be duplicated here.
- Keyless invitation/DM **operator authorization** is in scope for this
  workstream as a bounded requirement from the current goal: when an
  invitation or DM action is not already operator-authorized, the Agent routes
  a clear authorization request to its configured operator. See the
  [source audit](STABLE-2.0-SOURCE-AUDIT.md) for the exact missing behavior
  boundary. The broader keyless DM trust-management surface, Runtime Event UI,
  encrypted remote runtime-output streaming, reactions, Hermes, Gemini, ACP,
  and plaintext DMs remain separate product work.
- Broad refactoring and test-suite expansion are not readiness work. Add only
  tests that guard a changed boundary or a release acceptance failure.

## Branch Discipline

- Keep one commit or review unit per gap; do not mix Agent code, release
  operations, and cloud-template work.
- Treat the release contract as canonical. Update this backlog with status and
  evidence links rather than copying new acceptance rules into multiple files.
- Preserve existing `1.2.0` user state during upgrade: configuration, profile,
  memory, workspace, keys, message history, and supported session references.
- Do not publish stable artifacts from this branch.
