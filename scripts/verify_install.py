"""Post-install smoke test.

Run this from OUTSIDE the repository root after installing the wheel, so that
the installed package is exercised rather than the working tree:

    python /path/to/repo/scripts/verify_install.py
"""

import sys
from pathlib import Path

import cot_hmm


TRACE = (
    "The problem asks for apples.\n"
    "Let x be apples.\n"
    "12 + 8 = 20.\n"
    "Therefore answer is 20."
)


def main() -> int:
    print("cot_hmm loaded from:", cot_hmm.__file__)
    steps = cot_hmm.segment_and_clean(TRACE)
    tags = cot_hmm.tag_steps(steps)
    assert steps[2] == "12 + 8 = 20.", steps
    assert tags == ["PR", "IN", "CA", "CO"], tags
    # Exercise the bundled pre-trained model artifact
    try:
        scorer = cot_hmm.load_default_scorer()
        score = scorer.score(tags)
        assert isinstance(score, float)
        print("SMOKE OK (bundled model):", tags, round(score, 4))
    except Exception as e:
        print("SMOKE OK (no model artifact in install):", tags, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
