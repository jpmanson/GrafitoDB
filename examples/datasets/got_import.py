import argparse
import os
import time

from grafito import GrafitoDatabase


def _print_stats(db: GrafitoDatabase) -> None:
    labels = db.get_all_labels()
    rel_types = db.get_all_relationship_types()

    print("\n=== GOT Import Stats ===")
    print(f"Nodes: {db.get_node_count()}")
    for label in labels:
        print(f"  :{label} -> {db.get_node_count(label)}")
    print(f"Relationships: {db.get_relationship_count()}")
    for rel_type in rel_types:
        print(f"  :{rel_type} -> {db.get_relationship_count(rel_type)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Game of Thrones data via Cypher script.")
    parser.add_argument("--db", default="got.db", help="SQLite database file path")
    parser.add_argument(
        "--cypher",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "got-import.cypher"),
        help="Path to the .cypher script",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the database file before importing",
    )
    parser.add_argument(
        "--transaction",
        choices=("none", "statement", "script"),
        default="statement",
        help="Transaction scope for the import (none, statement, script)",
    )
    parser.add_argument(
        "--apoc-cache-dir",
        default=os.path.join(".grafito", "cache", "apoc"),
        help="Cache directory for apoc.load.jsonArray HTTP downloads (set empty to disable)",
    )
    args = parser.parse_args()

    if args.clean and args.db != ":memory:" and os.path.exists(args.db):
        try:
            os.remove(args.db)
        except PermissionError as exc:
            print(f"Cannot remove database file (locked?): {args.db}")
            print(f"Details: {exc}")
            return

    if not os.path.exists(args.cypher):
        print(f"Cypher script not found: {args.cypher}")
        return

    if args.apoc_cache_dir:
        os.environ["GRAFITO_APOC_CACHE_DIR"] = args.apoc_cache_dir
    else:
        os.environ.pop("GRAFITO_APOC_CACHE_DIR", None)

    db = GrafitoDatabase(args.db)
    try:
        with open(args.cypher, "r", encoding="utf-8") as handle:
            script = handle.read()
        statements = db._split_cypher_statements(script)
        total = len(statements)
        start = time.time()

        if args.transaction == "script":
            db.begin_transaction()
        try:
            for idx, statement in enumerate(statements, start=1):
                if not statement.strip():
                    continue
                preview = " ".join(statement.strip().split())[:120]
                print(f"[{idx}/{total}] {preview}")
                if args.transaction == "statement":
                    db.begin_transaction()
                db.execute(statement)
                if args.transaction == "statement":
                    db.commit()
            if args.transaction == "script":
                db.commit()
        except Exception:
            if args.transaction != "none":
                db.rollback()
            raise

        elapsed = time.time() - start
        print(f"\nImport completed in {elapsed:.2f}s")
        _print_stats(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
