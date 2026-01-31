# Example: Company Structure

This example models an organization with departments, employees, reporting
lines, skills, and management hierarchy.

Source: `examples/company_structure.py`.

## Data Model

Nodes:

- `(:Company {name, founded, industry, size})`
- `(:Department {name, budget})`
- `(:Person:Employee {name, title, email, hire_date, salary})`
- `(:Skill {name})`

Relationships:

- `(:Department)-[:PART_OF]->(:Company)`
- `(:Employee)-[:WORKS_IN {since}]->(:Department)`
- `(:Employee)-[:WORKS_AT {since}]->(:Company)`
- `(:Employee)-[:REPORTS_TO]->(:Employee)`
- `(:Department)-[:MANAGES]->(:Department)`
- `(:Employee)-[:HAS_SKILL {years}]->(:Skill)`

## Build the Graph

```python
from grafito import GrafitoDatabase

db = GrafitoDatabase(":memory:")

techcorp = db.create_node(
    labels=["Company"],
    properties={"name": "TechCorp", "founded": 2010, "industry": "Technology", "size": "large"},
)

engineering = db.create_node(labels=["Department"], properties={"name": "Engineering", "budget": 5_000_000})

db.create_relationship(engineering.id, techcorp.id, "PART_OF")
```

## Queries

### Organizational Hierarchy

```python
ceo_reports = db.get_neighbors(ceo.id, direction="incoming", rel_type="REPORTS_TO")
```

### Department Headcount

```python
eng_employees = db.get_neighbors(engineering.id, direction="incoming", rel_type="WORKS_IN")
```

### Employees with a Skill

```python
python_rels = db.match_relationships(target_id=python_skill.id, rel_type="HAS_SKILL")
```

### Reporting Chain

```python
current = engineer1
while True:
    managers = db.get_neighbors(current.id, direction="outgoing", rel_type="REPORTS_TO")
    if not managers:
        break
    current = managers[0]
```

## Try It

Run the full example:

```bash
python examples/company_structure.py
```
