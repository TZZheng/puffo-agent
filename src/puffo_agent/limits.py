"""Shared message-size limits imported by the message runtime."""

# Messages above this size are represented by a placeholder in prompts.
MAX_INLINE_MESSAGE_CHARS = 16000

# Page size used when reading a redacted message body.
MESSAGE_SEGMENT_CHARS = 8000

# Older catch-up messages are stored but skip the model.
DEFAULT_CATCHUP_STALE_HOURS = 48.0
