"""Shared skill-id rule — single definition, imported everywhere.

Also spelled as a ``CHECK`` constraint on ``skill_templates`` in puffo-server
migration 038. That copy can't import this one and can't be edited either
(sqlx checksums applied migrations), so the two are kept in step by hand.
"""

import re

# Both the catalog id and the ``.claude/skills/<id>/`` folder name, so a
# malformed id can't escape into a stray directory write. ``\Z`` not ``$``:
# Python's ``$`` also matches before a trailing newline, so ``"ok-id\n"``
# would pass here and be rejected by the SQL copy.
SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}\Z")
