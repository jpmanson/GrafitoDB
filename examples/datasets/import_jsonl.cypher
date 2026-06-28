WITH "examples/datasets/people.jsonl" AS url
CALL apoc.import.json(url) YIELD nodes, relationships
RETURN nodes, relationships;

MATCH (a:Person)-[r:KNOWS]->(b:Person)
RETURN a.name AS a, b.name AS b, r.since AS since;
