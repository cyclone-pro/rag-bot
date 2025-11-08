"""Backward-compatible entrypoint for the recruiter brain CLI."""
from recruiterbrain.app_workflow import answer_question, llm_plan, print_help, run_cli

__all__ = ["answer_question", "llm_plan", "print_help", "run_cli"]

if __name__ == "__main__":
    run_cli()
