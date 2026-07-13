"""Default: print help for subcommands."""

import sys

HELP = """Path A blocks package.

  python -m training.path_a_blocks.diagnostics [args]
  python -m training.path_a_blocks.run_1a [args]
  python -m training.path_a_blocks.run_1b [args]
  python -m training.path_a_blocks.run_1c [args]
  python -m training.path_a_blocks.build_minimal_ranks [args]
  python -m training.path_a_blocks.run_wrap --exp paid_only [args]
  python -m training.path_a_blocks.run_wrap --all [args]
"""


def main() -> int:
    print(HELP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
