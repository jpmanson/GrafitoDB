# Example: Social Network

This example builds a small social graph with users, interests, friendships,
followers, and simple recommendations.

Source: `examples/social_network.py`.

## Data Model

Nodes:

- `(:Person:User {username, name, age, city, joined})`
- `(:Topic {name, category})`

Relationships:

- `(:User)-[:FRIENDS_WITH {since}]->(:User)` (bidirectional)
- `(:User)-[:FOLLOWS]->(:User)`
- `(:User)-[:INTERESTED_IN {level}]->(:Topic)`

## Build the Graph

```python
from grafito import GrafitoDatabase

db = GrafitoDatabase(":memory:")

alice = db.create_node(
    labels=["Person", "User"],
    properties={
        "username": "alice_wonder",
        "name": "Alice",
        "age": 28,
        "city": "San Francisco",
        "joined": "2020-01-15",
    },
)

python_topic = db.create_node(
    labels=["Topic"],
    properties={"name": "Python", "category": "Programming"},
)

db.create_relationship(alice.id, python_topic.id, "INTERESTED_IN", {"level": "expert"})
```

## Queries

### Friends of a User

```python
friends = db.get_neighbors(alice.id, direction="outgoing", rel_type="FRIENDS_WITH")
```

### Users Interested in a Topic

```python
python_rels = db.match_relationships(target_id=python_topic.id, rel_type="INTERESTED_IN")
users = [db.get_node(rel.source_id) for rel in python_rels]
```

### Shortest Path Between Users

```python
path = db.find_shortest_path(alice.id, emma.id)
```

### Followers of a User

```python
followers = db.get_neighbors(alice.id, direction="incoming", rel_type="FOLLOWS")
```

### Friend Recommendations (Friends of Friends)

```python
alice_friends = db.get_neighbors(alice.id, direction="outgoing", rel_type="FRIENDS_WITH")
known = {f.id for f in alice_friends}
known.add(alice.id)

potential = set()
for friend in alice_friends:
    for fof in db.get_neighbors(friend.id, direction="outgoing", rel_type="FRIENDS_WITH"):
        if fof.id not in known:
            potential.add(fof.id)
```

## Try It

Run the full example:

```bash
python examples/social_network.py
```
