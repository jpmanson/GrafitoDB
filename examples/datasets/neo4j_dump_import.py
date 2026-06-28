import argparse
import orjson
import os
import time

from grafito import GrafitoDatabase


def _build_summary(db: GrafitoDatabase) -> dict:
    labels = db.get_all_labels()
    rel_types = db.get_all_relationship_types()
    label_samples = {}
    rel_samples = {}
    rel_with_props = {}
    for label in labels:
        row = db.conn.execute(
            """
            SELECT n.properties
            FROM nodes n
            JOIN node_labels nl ON nl.node_id = n.id
            JOIN labels l ON l.id = nl.label_id
            WHERE l.name = ?
            LIMIT 1
            """,
            (label,),
        ).fetchone()
        if row:
            try:
                props = orjson.loads(row["properties"] or "{}")
            except Exception:
                props = {}
            label_samples[label] = list(props.keys())[:3]
    for rel_type in rel_types:
        row = db.conn.execute(
            "SELECT properties FROM relationships WHERE type = ? LIMIT 1",
            (rel_type,),
        ).fetchone()
        if row:
            try:
                props = orjson.loads(row["properties"] or "{}")
            except Exception:
                props = {}
            rel_samples[rel_type] = list(props.keys())[:3]
        count_row = db.conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM relationships
            WHERE type = ?
              AND properties IS NOT NULL
              AND properties != '{}'
            """,
            (rel_type,),
        ).fetchone()
        rel_with_props[rel_type] = count_row["count"] if count_row else 0
    return {
        "nodes_total": db.get_node_count(),
        "relationships_total": db.get_relationship_count(),
        "labels": {label: db.get_node_count(label) for label in labels},
        "relationship_types": {rel_type: db.get_relationship_count(rel_type) for rel_type in rel_types},
        "label_property_samples": label_samples,
        "relationship_property_samples": rel_samples,
        "relationship_properties_non_empty": rel_with_props,
    }


def _print_summary(summary: dict) -> None:
    print("\n=== Neo4j Dump Import Summary ===")
    print(f"Nodes: {summary['nodes_total']}")
    for label, count in summary["labels"].items():
        print(f"  :{label} -> {count}")
    print(f"Relationships: {summary['relationships_total']}")
    for rel_type, count in summary["relationship_types"].items():
        print(f"  :{rel_type} -> {count}")
    if summary.get("label_property_samples"):
        print("\nSample node properties:")
        for label, props in summary["label_property_samples"].items():
            print(f"  :{label} -> {props}")
    if summary.get("relationship_property_samples"):
        print("\nSample relationship properties:")
        for rel_type, props in summary["relationship_property_samples"].items():
            print(f"  :{rel_type} -> {props}")
    if summary.get("relationship_properties_non_empty"):
        print("\nRelationships with non-empty properties:")
        for rel_type, count in summary["relationship_properties_non_empty"].items():
            print(f"  :{rel_type} -> {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a Neo4j .dump into an in-memory Grafito DB.")
    parser.add_argument(
        "--dump",
        help="Absolute path to a Neo4j .dump file.",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path to write the summary JSON output.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N nodes/relationships (0 to disable).",
    )
    parser.add_argument(
        "--node-limit",
        type=int,
        help="Optional maximum number of nodes to import.",
    )
    parser.add_argument(
        "--rel-limit",
        type=int,
        help="Optional maximum number of relationships to import.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the extracted dump directory (useful for debugging Windows locks).",
    )
    parser.add_argument(
        "--temp-dir",
        help="Optional directory to extract the dump into (avoids system temp).",
    )
    args = parser.parse_args()

    if not args.dump:
        parser.print_help()
        return 2

    dump_path = os.path.abspath(args.dump)
    if not os.path.exists(dump_path):
        print(f"Dump file not found: {dump_path}")
        return 1

    db = GrafitoDatabase(":memory:")
    start = time.time()
    try:
        print(f"Importing {dump_path} into :memory: ...")
        progress_every = args.progress_every if args.progress_every and args.progress_every > 0 else None
        db.import_neo4j_dump(
            dump_path,
            progress_every=progress_every,
            node_limit=args.node_limit,
            rel_limit=args.rel_limit,
            cleanup=not args.keep_temp,
            temp_dir=args.temp_dir,
        )
    except Exception as exc:
        print(f"Import failed: {exc}")
        return 1
    finally:
        elapsed = time.time() - start
        print(f"Import completed in {elapsed:.2f}s")

    summary = _build_summary(db)
    _print_summary(summary)
    if args.summary_json:
        try:
            with open(args.summary_json, "w", encoding="utf-8") as handle:
                handle.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode("utf-8"))
            print(f"\nSummary written to {args.summary_json}")
        except OSError as exc:
            print(f"\nFailed to write summary JSON: {exc}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
