"""Allow ``python -m lit_search.cli`` without installing the console script."""

import sys

from lit_search.cli.main import main

if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
