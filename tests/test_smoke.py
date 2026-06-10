"""
Headless smoke test: render every dashboard page via Streamlit's AppTest and
assert no page raises. Also exercises the MCDM sliders to confirm live
re-ranking does not error.

Run:  uv run python tests/test_smoke.py
      (or: uv run pytest tests/test_smoke.py)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
PAGES = [
    "Project Overview",
    "Student Profile",
    "Concept Dependency Graph",
    "Learning Path Recommendation",
    "RL Policy",
    "Mastery Progress Analytics",
    "MCDM Decision Settings",
    "Performance Metrics",
]


def _run_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, f"Initial run raised: {at.exception}"
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"Page '{page}' raised: {at.exception}"
    return at


def test_all_pages():
    for page in PAGES:
        _run_page(page)
        print(f"  [ok] {page}")


def test_mcdm_live_rerank():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("MCDM Decision Settings").run()
    assert not at.exception
    # Crank "Difficulty" weight to the max and confirm no error on re-rank.
    at.slider(key="w_Difficulty").set_value(1.0).run()
    assert not at.exception, f"MCDM re-rank raised: {at.exception}"
    print("  [ok] MCDM live re-rank")


if __name__ == "__main__":
    print("Smoke-testing all pages:")
    test_all_pages()
    test_mcdm_live_rerank()
    print("All pages rendered without exceptions.")
