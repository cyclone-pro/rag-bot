"""Console entry point for `python -m recruiterbrain`."""
from __future__ import annotations

import argparse
import sys
from typing import Sequence
from app_workflow import answer_question, run_cli


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recruiterbrain",
        description="Ask recruiting questions over the Milvus candidate pool.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="If supplied, answer a single question and exit. Otherwise start the interactive shell.",
    )
    args = parser.parse_args(argv)

    if args.question:
        try:
            print(answer_question(args.question))
        except Exception as exc:  # pragma: no cover - CLI feedback only
            parser.exit(status=1, message=f"Error: {exc}\n")
        return 0

    run_cli()
    return 0


if __name__ == "__main__":
    sys.exit(main())
