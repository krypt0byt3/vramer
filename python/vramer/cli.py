"""Unified ``python -m vramer`` dispatcher.

Subcommands map to the per-tool CLIs:

    python -m vramer plan   --profile ... --model ...
    python -m vramer detect [--json]
    python -m vramer score  --profile ... --model ...

Each subcommand forwards its remaining arguments to the matching tool's
``main()`` (the same entry points installed as ``vramer-plan`` / ``vramer-detect``
/ ``vramer-score``).
"""

from __future__ import annotations

import sys
from typing import List, Optional

from . import __version__, detect, planner, score

_COMMANDS = {
    "plan": planner.main,
    "detect": detect.main,
    "score": score.main,
}


def _usage() -> str:
    return (
        "usage: python -m vramer <command> [options]\n\n"
        "commands:\n"
        "  plan     estimate decode throughput (vramer-plan)\n"
        "  detect   emit a hardware profile    (vramer-detect)\n"
        "  score    recommend a config         (vramer-score)\n\n"
        "Run 'python -m vramer <command> --help' for command options.\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        sys.stdout.write(_usage())
        return 0

    if argv[0] in ("-V", "--version", "version"):
        print(f"vramer {__version__}")
        return 0

    cmd, rest = argv[0], argv[1:]
    handler = _COMMANDS.get(cmd)
    if handler is None:
        sys.stderr.write(f"vramer: unknown command {cmd!r}\n\n")
        sys.stderr.write(_usage())
        return 2

    handler(rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
