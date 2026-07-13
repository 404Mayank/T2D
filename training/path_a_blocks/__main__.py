"""Default: print help for subcommands."""

import sys

HELP = """Path A blocks package.

  python -m training.path_a_blocks.diagnostics [args]
  python -m training.path_a_blocks.run_1a [args]
"""


def main() -> int:
    print(HELP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
