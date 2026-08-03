# One-shot local reminder contract

The Agent owns one-shot reminder intent, occurrence identity, plaintext
content, cancellation, delivery, and local scheduling in its per-Agent
`messages.db`. This contract is intentionally provider-neutral.

## Tools

- `create_reminder(content, target, intended_at)` requires exact non-empty
  content, a canonical target (`dm:<peer>`, `channel:<space>:<channel>`, or a
  channel thread target), and an RFC3339 timestamp with an explicit offset.
- `list_reminders(state="", limit=50)` returns reminders ordered by intended
  time and occurrence identity.
- `cancel_reminder(reminder_id)` is idempotent. It prevents an undelivered
  occurrence from firing and never removes a delivered local Inbox event.

Each create/cancel result is the same structured object containing immutable
`reminder_id`, `occurrence_id`, target, exact content, intended UTC time, and
the actual-fire, creation, cancellation, and delivery facts when present.

## Durable delivery

`reminder_occurrences` stores one immutable intent plus its sole occurrence.
The state transition is:

```text
scheduled -> claimed -> delivered
scheduled/claimed -> cancelled
```

Claiming records the actual fire time once. Delivery creates
`reminder-occurrence:<occurrence_id>` as a `local_runtime` pending Inbox row
and transitions the occurrence to `delivered` in the same SQLite transaction.
Duplicate ticks and restart recovery therefore find either the durable claim
or the committed terminal event; they cannot enqueue a second event. A late
occurrence remains a fact with its intended and actual times, not an instruction
to act, skip, reply, apologize, or stay silent.

The local event contains:

```json
{
  "event_type": "reminder",
  "reminder_id": "...",
  "occurrence_id": "...",
  "target": "channel:space:channel",
  "content": "exact Agent-authored content",
  "intended_at": "UTC RFC3339",
  "actual_fire_at": "UTC RFC3339"
}
```

It uses the existing local-order frontier and `GlobalInboxRuntime.notify()`
path. Consequently a target with an ordinary message and a reminder has one
content-free Inbox notice, one ordered `read_inbox` page, and one ordinary turn
path. Message records retain their own timestamps and message projection.

## Restart and ownership boundary

The scheduler reopens the same local SQLite file, delivers durable claimed
work before waiting, and waits only until the next durable deadline or a
create/cancel signal. This covers an online local Agent and a local worker that
restarts later.

The next Server contract extends this local behavior so an offline Agent can
catch up and reconstruct local reminder state. The Agent encrypts the private
payload with its remote-data DEK (the current Core architecture calls this
`MessageBackupDEK`) directly through AEAD with a unique nonce. It does not
derive a Reminder-specific key and does not reuse the local SQLCipher
`DatabaseDek`. The Server receives only opaque ciphertext plus the minimum
plaintext occurrence, lifecycle, and scheduling metadata needed to index and
wake due work.

Cross-device timer election and transfer remain outside the first Server
snapshot contract.

## Non-goals

This slice has no editing, rescheduling, recurrence, snooze, browser surface,
cloud sandbox lifecycle scheduling, cross-device election, or
provider-specific scheduler behavior. Server storage/API and Agent snapshot
reconciliation are delivered as a separate reviewed slice.
