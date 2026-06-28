from pathlib import Path

from grafito import GrafitoDatabase


JSONL_PATH = Path(__file__).with_name("people.jsonl")


def main() -> None:
    db = GrafitoDatabase(":memory:")
    result = db.execute(f"""
        WITH '{JSONL_PATH.as_posix()}' AS url
        CALL apoc.import.json(url) YIELD nodes, relationships
        RETURN nodes, relationships
    """)
    print(result)

    matches = db.execute("""
        MATCH (a:Person)-[r:KNOWS]->(b:Person)
        RETURN a.name AS a, b.name AS b, r.since AS since
    """)
    print(matches)
    db.close()


if __name__ == "__main__":
    main()
