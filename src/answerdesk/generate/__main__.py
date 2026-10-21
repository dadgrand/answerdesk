"""``python -m answerdesk.generate --stamp`` — prompt-registry authoring entry point.

Separate from :mod:`answerdesk.generate.prompts` so that running it produces no
"module found in sys.modules" warning: the package imports ``prompts`` on first import.

    # AT-GEN-04
"""

from __future__ import annotations

from answerdesk.generate.prompts import main

if __name__ == "__main__":
    raise SystemExit(main())
