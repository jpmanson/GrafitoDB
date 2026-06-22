"""Import/export an Open Knowledge Format (OKF) bundle with GrafitoDB.

OKF is a directory of markdown files with YAML frontmatter.
This example imports the sample bundle in ``examples/okf_bundle``, runs a Cypher
query and a full-text search, then exports the graph back to a new OKF bundle.
"""

import tempfile
from pathlib import Path

from grafito import GrafitoDatabase

BUNDLE = Path(__file__).parent / "okf_bundle"


def main() -> None:
    db = GrafitoDatabase(":memory:")
    summary = db.import_okf_bundle(str(BUNDLE))
    print(f"Imported: {summary}")

    # Cypher: which concepts does the Orders table link to?
    print("\nOutgoing links from 'Orders':")
    rows = db.execute(
        """
        MATCH (a {title: 'Orders'})-[r:LINKS_TO]->(b)
        RETURN b.title AS target, r.anchor AS anchor
        ORDER BY target
        """
    )
    for row in rows:
        print(f"  Orders -> {row['target']} (anchor: {row['anchor']!r})")

    # Full-text search over the imported bodies/descriptions.
    if db.has_fts5():
        print("\nFull-text search for 'customer':")
        for hit in db.text_search("customer", k=5):
            node = hit["entity"]
            print(f"  {node.properties.get('title')} (score={hit['score']:.3f})")

    # Export the graph back to a fresh OKF bundle (round-trip), including a
    # self-contained viz.html graph viewer.
    out_dir = Path(tempfile.mkdtemp(prefix="okf_export_"))
    export_summary = db.export_okf_bundle(str(out_dir), write_viz=True)
    print(f"\nExported {export_summary} to {out_dir}")
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out_dir)}")

    db.close()


if __name__ == "__main__":
    main()
