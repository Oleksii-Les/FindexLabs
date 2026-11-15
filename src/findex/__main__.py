"""``python -m findex`` is the same as the ``findex`` command.

The guard is not decoration: with the ``spawn`` start method every worker process re-imports
the main module. Without ``if __name__ == "__main__"`` each worker would start the CLI, which
would start more workers, and so on.
"""

from findex.cli import app

if __name__ == "__main__":
    app(prog_name="findex")
