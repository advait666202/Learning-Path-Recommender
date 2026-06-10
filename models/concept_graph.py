"""
Concept dependency graph (NetworkX).

Builds a directed acyclic graph from the canonical catalogue in
:mod:`utils.concepts`, where an edge ``prereq -> concept`` means ``prereq`` must
be completed before ``concept``. Node attributes carry everything the dashboard
needs to render a node tooltip: name, difficulty, prerequisites, track, and the
average mastery observed in the dataset (when ``data/students.csv`` exists).

Persistence uses :mod:`pickle` (NetworkX removed ``write_gpickle`` in 3.0) but
keeps the conventional ``.gpickle`` extension.

Run directly to (re)build and save the graph:
    uv run python models/concept_graph.py
"""

from __future__ import annotations

import os
import pickle
import sys

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import pandas as pd

from utils import concepts as C
from utils.config import DATA_DIR, GRAPH_PICKLE, STUDENTS_CSV


def _avg_mastery_by_concept() -> dict[str, float]:
    """Mean Current_Mastery per concept name from the dataset (empty if missing)."""
    if not os.path.exists(STUDENTS_CSV):
        return {}
    df = pd.read_csv(STUDENTS_CSV)
    return df.groupby("Concept_Current")["Current_Mastery"].mean().to_dict()


def build_graph() -> nx.DiGraph:
    """Construct the concept DiGraph with rich node attributes."""
    avg_mastery = _avg_mastery_by_concept()
    g = nx.DiGraph()

    for c in C.CONCEPTS:
        g.add_node(
            c["id"],
            name=c["name"],
            difficulty=c["difficulty"],
            prerequisites=list(c["prerequisites"]),
            track=c["track"],
            avg_mastery=round(float(avg_mastery.get(c["name"], 0.0)), 3),
        )

    for c in C.CONCEPTS:
        for p in c["prerequisites"]:
            g.add_edge(p, c["id"])

    if not nx.is_directed_acyclic_graph(g):
        raise RuntimeError("Concept graph contains a cycle; expected a DAG.")
    return g


def save_graph(g: nx.DiGraph, path: str = GRAPH_PICKLE) -> None:
    """Pickle the graph to ``path``."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: str = GRAPH_PICKLE) -> nx.DiGraph:
    """Load the pickled graph, rebuilding it on the fly if the file is missing."""
    if not os.path.exists(path):
        g = build_graph()
        save_graph(g, path)
        return g
    with open(path, "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    graph = build_graph()
    save_graph(graph)
    print(f"Built concept graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges.")
    print("Is DAG:", nx.is_directed_acyclic_graph(graph))
    print("Topological order:",
          [C.concept_name(n) for n in nx.topological_sort(graph)])
    print(f"Saved to {GRAPH_PICKLE}")
