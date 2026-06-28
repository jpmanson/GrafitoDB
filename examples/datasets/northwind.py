import argparse
import os
import sys

from grafito.database import GrafitoDatabase


def _scalar(db, query, key):
    results = db.execute(query)
    if not results:
        raise AssertionError(f"No results for query: {query}")
    return results[0].get(key)


def _assert_equal(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def _assert_isinstance(label, value, expected_type):
    if not isinstance(value, expected_type):
        raise AssertionError(f"{label}: expected {expected_type.__name__}, got {type(value).__name__}")


def _run_checks(db):
    expected_counts = {
        "Product": 77,
        "Category": 8,
        "Supplier": 29,
        "Customer": 91,
        "Order": 830,
    }
    for label, expected in expected_counts.items():
        count = _scalar(db, f"MATCH (n:{label}) RETURN COUNT(n) AS count", "count")
        _assert_equal(f"{label} count", count, expected)

    expected_rels = {
        "PART_OF": 77,
        "SUPPLIES": 77,
        "PURCHASED": 830,
        "ORDERS": 2155,
    }
    for rel_type, expected in expected_rels.items():
        count = _scalar(db, f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(r) AS count", "count")
        _assert_equal(f"{rel_type} count", count, expected)

    price = _scalar(
        db,
        "MATCH (p:Product {productID: '1'}) RETURN p.unitPrice AS value",
        "value",
    )
    _assert_isinstance("Product.unitPrice", price, float)

    discontinued = _scalar(
        db,
        "MATCH (p:Product {productID: '1'}) RETURN p.discontinued AS value",
        "value",
    )
    _assert_isinstance("Product.discontinued", discontinued, bool)

    quantity = _scalar(
        db,
        "MATCH ()-[d:ORDERS]->() RETURN d.quantity AS value LIMIT 1",
        "value",
    )
    _assert_isinstance("ORDERS.quantity", quantity, int)


def _print_stats(db):
    print("Northwind data stats:", flush=True)
    stats_queries = [
        ("Nodes total", "MATCH (n) RETURN COUNT(n) AS value"),
        ("Relationships total", "MATCH ()-[r]->() RETURN COUNT(r) AS value"),
        ("Products", "MATCH (n:Product) RETURN COUNT(n) AS value"),
        ("Categories", "MATCH (n:Category) RETURN COUNT(n) AS value"),
        ("Suppliers", "MATCH (n:Supplier) RETURN COUNT(n) AS value"),
        ("Customers", "MATCH (n:Customer) RETURN COUNT(n) AS value"),
        ("Orders", "MATCH (n:Order) RETURN COUNT(n) AS value"),
        ("PART_OF", "MATCH ()-[r:PART_OF]->() RETURN COUNT(r) AS value"),
        ("SUPPLIES", "MATCH ()-[r:SUPPLIES]->() RETURN COUNT(r) AS value"),
        ("PURCHASED", "MATCH ()-[r:PURCHASED]->() RETURN COUNT(r) AS value"),
        ("ORDERS", "MATCH ()-[r:ORDERS]->() RETURN COUNT(r) AS value"),
    ]
    for label, query in stats_queries:
        value = _scalar(db, query, "value")
        print(f"- {label}: {value}", flush=True)

    # Print all labels and relationship types
    all_labels = db.get_all_labels()
    print(f"All labels in database: {all_labels}", flush=True)

    all_rel_types = db.get_all_relationship_types()
    print(f"All relationship types: {all_rel_types}", flush=True)


def _load_script(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        script = handle.read()
    db = GrafitoDatabase(":memory:")
    try:
        return db._split_cypher_statements(script)
    finally:
        db.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Load Northwind and run data checks.")
    parser.add_argument(
        "--script",
        default=os.path.join(script_dir, "northwind.cypher"),
        help="Path to the Northwind Cypher script.",
    )
    parser.add_argument(
        "--db",
        default=":memory:",
        help="SQLite database path (default: :memory:).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the database file before loading (ignored for :memory:).",
    )
    parser.add_argument(
        "--sql-trace",
        action="store_true",
        help="Print every SQLite statement executed.",
    )
    args = parser.parse_args()

    script_path = args.script
    if not os.path.exists(script_path) and not os.path.isabs(script_path):
        candidate = os.path.join(script_dir, script_path)
        if os.path.exists(candidate):
            script_path = candidate

    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return 1

    if args.clean and args.db != ":memory:" and os.path.exists(args.db):
        try:
            os.remove(args.db)
        except PermissionError:
            print(f"Cannot remove '{args.db}': file is locked by another process.", flush=True)
            return 1

    db = GrafitoDatabase(args.db)
    if args.sql_trace:
        db.conn.set_trace_callback(lambda stmt: print(f"[SQL] {stmt}", flush=True))
    try:
        existing = _scalar(db, "MATCH (n) RETURN COUNT(n) AS count", "count")
        if existing and args.db != ":memory:":
            print("Existing data detected, skipping load.", flush=True)
            _run_checks(db)
            print("Northwind checks passed on existing data.")
            _print_stats(db)
            return 0

        statements = _load_script(script_path)
        constraint_statements = statements[:5]
        data_statements = statements[5:]

        for index, statement in enumerate(constraint_statements, start=1):
            trimmed = " ".join(statement.strip().split())
            preview = trimmed[:120] + ("..." if len(trimmed) > 120 else "")
            print(f"[{index}/{len(statements)}] {preview}", flush=True)
            db.execute(statement)

        # Load data in a single transaction for file-backed DBs.
        if data_statements:
            db.begin_transaction()
            try:
                for index, statement in enumerate(data_statements, start=6):
                    trimmed = " ".join(statement.strip().split())
                    preview = trimmed[:120] + ("..." if len(trimmed) > 120 else "")
                    print(f"[{index}/{len(statements)}] {preview}", flush=True)
                    db.execute(statement)
                db.commit()
            except Exception:
                db.rollback()
                raise
        _run_checks(db)
        _print_stats(db)
    except Exception as exc:
        print(f"Northwind checks failed: {exc}")
        return 1
    finally:
        db.close()

    print("Northwind load and checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
