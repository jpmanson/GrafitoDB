"""Import/query/export an OKF bundle with the high-level OKFBundle API.

OKF is a directory of markdown files with YAML frontmatter. ``OKFBundle`` loads
one into a queryable graph and speaks concepts/links/search instead of
nodes/relationships — with the full graph one attribute away (``kb.db``).

Run:  python examples/okf_import.py
"""

import tempfile
from pathlib import Path

from grafito.okf import OKFBundle

BUNDLE = Path(__file__).parent / "okf_bundle"


def main() -> None:
    # Load the bundle into an in-memory graph (full-text search configured).
    kb = OKFBundle.load(BUNDLE)
    print(f"Imported: {kb.summary}")

    # Concepts and their links, in OKF vocabulary (no Cypher needed).
    orders = kb.concept("tables/orders")
    print(f"\n'{orders.title}' ({orders.type}) links to:")
    for target in orders.links():
        print(f"  -> {target.title}")

    # Full-text search; results come back as a uniform Hit (concept/score/via).
    if kb.db.has_fts5():
        print("\nSearch for 'customer':")
        for hit in kb.search("customer", mode="text", k=5):
            print(f"  {hit.concept.title}  (score={hit.score:.3f})")

    # Escape hatch: the full graph is always available via kb.db / kb.execute.
    total = kb.execute("MATCH (n) RETURN count(n) AS n")[0]["n"]
    print(f"\nNodes in the graph (via Cypher escape hatch): {total}")

    # Export back to a fresh OKF bundle, with a self-contained viz.html.
    out_dir = Path(tempfile.mkdtemp(prefix="okf_export_"))
    print(f"\nExported {kb.save(out_dir, write_viz=True)} to {out_dir}")
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out_dir)}")

    kb.db.close()


if __name__ == "__main__":
    main()
