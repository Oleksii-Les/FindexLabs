"""The ONE place where logging is configured (called from entry points, never from libraries)."""

from __future__ import annotations

import logging
import sys


def configure_logging(verbosity: int = 0) -> None:
    """0 -> WARNING, 1 (-v) -> INFO, 2+ (-vv) -> DEBUG. Diagnostics always go to stderr,
    so stdout stays clean for data (``findex search --json | jq``)."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,  # replace handlers from an earlier call (tests, repeated invocations)
    )
