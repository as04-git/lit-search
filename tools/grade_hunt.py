"""Grade a finished hunt (or an Ai2 Paper Finder result) against a confidence-set case.

Usage:
    python tools/grade_hunt.py docs/confidence/cases.json CASE_ID HUNT_ID
    python tools/grade_hunt.py docs/confidence/cases.json CASE_ID paper-finder.json

The grading itself lives in :mod:`lit_search.evaluation.grading`; ``lit-search eval`` uses it too.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lit_search.evaluation.grading import (
    grade,
    load_case,
    paper_finder_items,
    render_grade,
    session_items,
)
from lit_search.hunt.session import HuntSession


def main(argv: list[str]) -> int:
    cases_path, case_id, run = argv
    case = load_case(Path(cases_path), case_id)
    if run.endswith(".json"):
        items = paper_finder_items(Path(run))
        header = f"Paper Finder {run} · {len(items)} results"
    else:
        session = HuntSession.load(run)
        items = session_items(session)
        order = (session.data.get("report") or {}).get("shortlist") or []
        header = (
            f"hunt {session.data['id']} · {len(session.data['moves'])} moves · "
            f"{len(session.candidates)} candidates · report: {'yes' if order else 'NOT RUN'}"
        )
    tiers, outside = grade(case, items)
    print(render_grade(f"{case_id} · {header}", tiers, outside))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
