"""
Canonical concept catalogue for the Personalized Learning Path Recommender.

THIS MODULE IS THE SINGLE SOURCE OF TRUTH for the concept set. The NetworkX
graph (models/concept_graph.py), the synthetic dataset (utils/data_gen.py) and
the RL environment (rl/environment.py) all import the concept list from here so
there is no drift between the three.

Each concept is a dict:
    id            int    - stable index used as the RL "current concept" state dim
    name          str    - human readable concept name
    difficulty    int    - 1..5 (1 = easiest, 5 = hardest)
    prerequisites list   - list of concept ids that must be completed first
    track         str    - thematic branch ("Foundations", "Programming",
                           "CS Theory", "Data/ML")

The list is intentionally a connected DAG: every concept is reachable from the
root ("Python Basics", id 0) and there are no cycles. A couple of branch edges
(e.g. Data Structures depends on both Functions and Recursion-adjacent ideas)
keep it from being a pure chain.
"""

from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# Canonical concept catalogue (24 concepts).
# ---------------------------------------------------------------------------
CONCEPTS: list[dict] = [
    {"id": 0,  "name": "Python Basics",        "difficulty": 1, "prerequisites": [],          "track": "Foundations"},
    {"id": 1,  "name": "Variables",            "difficulty": 1, "prerequisites": [0],         "track": "Foundations"},
    {"id": 2,  "name": "Data Types",           "difficulty": 1, "prerequisites": [1],         "track": "Foundations"},
    {"id": 3,  "name": "Operators",            "difficulty": 1, "prerequisites": [2],         "track": "Foundations"},
    {"id": 4,  "name": "Conditionals",         "difficulty": 2, "prerequisites": [3],         "track": "Programming"},
    {"id": 5,  "name": "Loops",                "difficulty": 2, "prerequisites": [4],         "track": "Programming"},
    {"id": 6,  "name": "Functions",            "difficulty": 2, "prerequisites": [5],         "track": "Programming"},
    {"id": 7,  "name": "Strings",              "difficulty": 2, "prerequisites": [2, 5],      "track": "Programming"},
    {"id": 8,  "name": "Lists & Collections",  "difficulty": 2, "prerequisites": [5],         "track": "Programming"},
    {"id": 9,  "name": "Dictionaries",         "difficulty": 3, "prerequisites": [8],         "track": "Programming"},
    {"id": 10, "name": "File I/O",             "difficulty": 3, "prerequisites": [6, 7],      "track": "Programming"},
    {"id": 11, "name": "Error Handling",       "difficulty": 3, "prerequisites": [6],         "track": "Programming"},
    {"id": 12, "name": "OOP",                  "difficulty": 3, "prerequisites": [6, 9],      "track": "Programming"},
    {"id": 13, "name": "Data Structures",      "difficulty": 4, "prerequisites": [8, 9],      "track": "CS Theory"},
    {"id": 14, "name": "Recursion",            "difficulty": 4, "prerequisites": [6],         "track": "CS Theory"},
    {"id": 15, "name": "Algorithms",           "difficulty": 4, "prerequisites": [13, 14],    "track": "CS Theory"},
    {"id": 16, "name": "Complexity Analysis",  "difficulty": 4, "prerequisites": [15],        "track": "CS Theory"},
    {"id": 17, "name": "NumPy",                "difficulty": 3, "prerequisites": [8, 12],     "track": "Data/ML"},
    {"id": 18, "name": "Pandas",               "difficulty": 3, "prerequisites": [17],        "track": "Data/ML"},
    {"id": 19, "name": "Data Visualization",   "difficulty": 3, "prerequisites": [18],        "track": "Data/ML"},
    {"id": 20, "name": "Statistics",           "difficulty": 4, "prerequisites": [17],        "track": "Data/ML"},
    {"id": 21, "name": "ML Basics",            "difficulty": 5, "prerequisites": [16, 18, 20], "track": "Data/ML"},
    {"id": 22, "name": "Neural Networks",      "difficulty": 5, "prerequisites": [21],        "track": "Data/ML"},
    {"id": 23, "name": "Deep Learning",        "difficulty": 5, "prerequisites": [22],        "track": "Data/ML"},
]

NUM_CONCEPTS: int = len(CONCEPTS)

# The "learning objective" terminal concept used for path-completion analytics.
OBJECTIVE_CONCEPT_ID: int = 23  # Deep Learning

# ---------------------------------------------------------------------------
# Derived lookup tables (built once at import time).
# ---------------------------------------------------------------------------
_BY_ID: dict[int, dict] = {c["id"]: c for c in CONCEPTS}
_NAME_BY_ID: dict[int, str] = {c["id"]: c["name"] for c in CONCEPTS}
_ID_BY_NAME: dict[str, int] = {c["name"]: c["id"] for c in CONCEPTS}


# ---------------------------------------------------------------------------
# Public helpers.
# ---------------------------------------------------------------------------
def concept_by_id(concept_id: int) -> dict:
    """Return the full concept dict for ``concept_id``."""
    return _BY_ID[concept_id]


def concept_name(concept_id: int) -> str:
    """Return the display name for ``concept_id``."""
    return _NAME_BY_ID[concept_id]


def concept_index(name: str) -> int:
    """Return the id for a concept ``name`` (inverse of :func:`concept_name`)."""
    return _ID_BY_NAME[name]


def difficulty_of(concept_id: int) -> int:
    """Return the 1..5 difficulty rating for ``concept_id``."""
    return _BY_ID[concept_id]["difficulty"]


def prerequisites_of(concept_id: int) -> list[int]:
    """Return the list of prerequisite concept ids for ``concept_id``."""
    return list(_BY_ID[concept_id]["prerequisites"])


def all_ids() -> list[int]:
    """Return all concept ids in canonical order."""
    return [c["id"] for c in CONCEPTS]


def all_names() -> list[str]:
    """Return all concept names in canonical order."""
    return [c["name"] for c in CONCEPTS]


def prerequisites_satisfied(concept_id: int, completed: Iterable[int]) -> bool:
    """True if every prerequisite of ``concept_id`` is in ``completed``."""
    completed_set = set(completed)
    return all(p in completed_set for p in _BY_ID[concept_id]["prerequisites"])


def valid_next_concepts(completed: Iterable[int]) -> list[int]:
    """
    Graph-valid candidate actions given the set of already-completed concepts.

    A concept is a valid next step when it is *not* already completed and *all*
    of its prerequisites are completed. This is the action mask used by both the
    RL agent and the MCDM re-ranker.
    """
    completed_set = set(completed)
    candidates = [
        c["id"]
        for c in CONCEPTS
        if c["id"] not in completed_set and prerequisites_satisfied(c["id"], completed_set)
    ]
    return candidates


def topological_order() -> list[int]:
    """
    Return a deterministic topological ordering of the concept DAG via Kahn's
    algorithm (ties broken by id). Used by the dataset generator and the RL
    environment as a monotone "progress" axis toward the objective.
    """
    indeg = {cid: len(prerequisites_of(cid)) for cid in all_ids()}
    succ: dict[int, list[int]] = {cid: [] for cid in all_ids()}
    for cid in all_ids():
        for p in prerequisites_of(cid):
            succ[p].append(cid)
    queue = sorted([cid for cid, d in indeg.items() if d == 0])
    order: list[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nxt in succ[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
        queue.sort()
    if len(order) != NUM_CONCEPTS:
        raise RuntimeError("Concept graph is not a DAG; topological sort failed.")
    return order


def topological_position() -> dict[int, int]:
    """Map each concept id to its 0-based position in the topological order."""
    return {cid: pos for pos, cid in enumerate(topological_order())}


def reachable_from_root() -> bool:
    """Sanity check used by tests: every concept is reachable from id 0 via prereqs."""
    completed: set[int] = set()
    progressed = True
    while progressed:
        progressed = False
        for cid in all_ids():
            if cid not in completed and prerequisites_satisfied(cid, completed):
                completed.add(cid)
                progressed = True
    return len(completed) == NUM_CONCEPTS


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    print(f"{NUM_CONCEPTS} canonical concepts.")
    print("All reachable from root:", reachable_from_root())
    print("Initial valid next concepts (nothing completed):", valid_next_concepts([]))
