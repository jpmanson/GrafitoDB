"""Social network example for Grafito graph database.

This example demonstrates a social network graph with:
- People and their connections
- Friendships and following relationships
- Interests and hobbies
- Path finding between users
"""

from grafito import GrafitoDatabase


def main():
    print("=== Social Network Example ===\n")

    # Initialize database
    db = GrafitoDatabase(':memory:')

    # =========================================================================
    # Create People
    # =========================================================================
    print("Creating social network users...\n")

    alice = db.create_node(
        labels=['Person', 'User'],
        properties={
            'username': 'alice_wonder',
            'name': 'Alice',
            'age': 28,
            'city': 'San Francisco',
            'joined': '2020-01-15'
        }
    )

    bob = db.create_node(
        labels=['Person', 'User'],
        properties={
            'username': 'bob_builder',
            'name': 'Bob',
            'age': 32,
            'city': 'New York',
            'joined': '2019-06-20'
        }
    )

    carol = db.create_node(
        labels=['Person', 'User'],
        properties={
            'username': 'carol_coder',
            'name': 'Carol',
            'age': 26,
            'city': 'San Francisco',
            'joined': '2021-03-10'
        }
    )

    david = db.create_node(
        labels=['Person', 'User'],
        properties={
            'username': 'david_dev',
            'name': 'David',
            'age': 35,
            'city': 'Seattle',
            'joined': '2018-11-05'
        }
    )

    emma = db.create_node(
        labels=['Person', 'User'],
        properties={
            'username': 'emma_engineer',
            'name': 'Emma',
            'age': 29,
            'city': 'New York',
            'joined': '2020-09-01'
        }
    )

    # =========================================================================
    # Create Interest Topics
    # =========================================================================
    print("Creating interest topics...\n")

    python_topic = db.create_node(
        labels=['Topic'],
        properties={'name': 'Python', 'category': 'Programming'}
    )

    ai_topic = db.create_node(
        labels=['Topic'],
        properties={'name': 'Artificial Intelligence', 'category': 'Technology'}
    )

    hiking_topic = db.create_node(
        labels=['Topic'],
        properties={'name': 'Hiking', 'category': 'Outdoors'}
    )

    photography_topic = db.create_node(
        labels=['Topic'],
        properties={'name': 'Photography', 'category': 'Arts'}
    )

    # =========================================================================
    # Create Social Connections
    # =========================================================================
    print("Creating social connections...\n")

    # Friendships (bidirectional)
    db.create_relationship(alice.id, bob.id, 'FRIENDS_WITH', {'since': '2020-02-01'})
    db.create_relationship(bob.id, alice.id, 'FRIENDS_WITH', {'since': '2020-02-01'})

    db.create_relationship(alice.id, carol.id, 'FRIENDS_WITH', {'since': '2021-04-15'})
    db.create_relationship(carol.id, alice.id, 'FRIENDS_WITH', {'since': '2021-04-15'})

    db.create_relationship(bob.id, david.id, 'FRIENDS_WITH', {'since': '2019-08-20'})
    db.create_relationship(david.id, bob.id, 'FRIENDS_WITH', {'since': '2019-08-20'})

    db.create_relationship(carol.id, emma.id, 'FRIENDS_WITH', {'since': '2021-06-01'})
    db.create_relationship(emma.id, carol.id, 'FRIENDS_WITH', {'since': '2021-06-01'})

    # Following (unidirectional)
    db.create_relationship(alice.id, david.id, 'FOLLOWS')
    db.create_relationship(emma.id, alice.id, 'FOLLOWS')
    db.create_relationship(david.id, emma.id, 'FOLLOWS')

    # Interests
    db.create_relationship(alice.id, python_topic.id, 'INTERESTED_IN', {'level': 'expert'})
    db.create_relationship(alice.id, ai_topic.id, 'INTERESTED_IN', {'level': 'intermediate'})
    db.create_relationship(alice.id, hiking_topic.id, 'INTERESTED_IN', {'level': 'beginner'})

    db.create_relationship(bob.id, python_topic.id, 'INTERESTED_IN', {'level': 'intermediate'})
    db.create_relationship(bob.id, photography_topic.id, 'INTERESTED_IN', {'level': 'advanced'})

    db.create_relationship(carol.id, python_topic.id, 'INTERESTED_IN', {'level': 'advanced'})
    db.create_relationship(carol.id, ai_topic.id, 'INTERESTED_IN', {'level': 'expert'})

    db.create_relationship(david.id, hiking_topic.id, 'INTERESTED_IN', {'level': 'expert'})
    db.create_relationship(david.id, photography_topic.id, 'INTERESTED_IN', {'level': 'intermediate'})

    db.create_relationship(emma.id, ai_topic.id, 'INTERESTED_IN', {'level': 'advanced'})
    db.create_relationship(emma.id, hiking_topic.id, 'INTERESTED_IN', {'level': 'intermediate'})

    # =========================================================================
    # Query 1: Find All Friends of Alice
    # =========================================================================
    print("=" * 60)
    print("Query 1: Alice's Friends")
    print("=" * 60)

    alice_friends = db.get_neighbors(alice.id, direction='outgoing', rel_type='FRIENDS_WITH')
    print(f"\nAlice has {len(alice_friends)} friends:")
    for friend in alice_friends:
        city = friend.properties['city']
        print(f"  - {friend.properties['name']} (@{friend.properties['username']}) from {city}")

    # =========================================================================
    # Query 2: Find Users Interested in Python
    # =========================================================================
    print("\n" + "=" * 60)
    print("Query 2: Python Enthusiasts")
    print("=" * 60)

    # Find all INTERESTED_IN relationships pointing to Python topic
    python_rels = db.match_relationships(target_id=python_topic.id, rel_type='INTERESTED_IN')
    print(f"\n{len(python_rels)} users interested in Python:")
    for rel in python_rels:
        user = db.get_node(rel.source_id)
        level = rel.properties.get('level', 'N/A')
        print(f"  - {user.properties['name']} (level: {level})")

    # =========================================================================
    # Query 3: Find Users in Same City with Common Interests
    # =========================================================================
    print("\n" + "=" * 60)
    print("Query 3: San Francisco Users with Common Interests")
    print("=" * 60)

    sf_users = db.match_nodes(labels=['User'], properties={'city': 'San Francisco'})
    print(f"\nFound {len(sf_users)} users in San Francisco:")

    for user in sf_users:
        interests = db.get_neighbors(user.id, direction='outgoing', rel_type='INTERESTED_IN')
        interest_names = [topic.properties['name'] for topic in interests]
        print(f"  - {user.properties['name']}: {', '.join(interest_names)}")

    # =========================================================================
    # Query 4: Find Connection Path Between Users
    # =========================================================================
    print("\n" + "=" * 60)
    print("Query 4: Connection Path from Alice to Emma")
    print("=" * 60)

    path = db.find_shortest_path(alice.id, emma.id)
    if path:
        print(f"\nShortest path ({len(path)} people, {len(path)-1} degrees of separation):")
        for i, node in enumerate(path):
            print(f"  {i+1}. {node.properties['name']} (@{node.properties['username']})")
            if i < len(path) - 1:
                # Find the relationship between this node and the next
                rels = db.match_relationships(source_id=node.id, target_id=path[i+1].id)
                if rels:
                    print(f"      └─ {rels[0].type} →")
    else:
        print("\nNo path found")

    # =========================================================================
    # Query 5: Find Who Follows Alice
    # =========================================================================
    print("\n" + "=" * 60)
    print("Query 5: Alice's Followers")
    print("=" * 60)

    followers = db.get_neighbors(alice.id, direction='incoming', rel_type='FOLLOWS')
    print(f"\nAlice has {len(followers)} followers:")
    for follower in followers:
        print(f"  - {follower.properties['name']} (@{follower.properties['username']})")

    # =========================================================================
    # Query 6: Recommend Friends (Friends of Friends)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Query 6: Friend Recommendations for Alice")
    print("=" * 60)

    # Get Alice's direct friends
    alice_friend_ids = {f.id for f in alice_friends}
    alice_friend_ids.add(alice.id)  # Don't recommend Alice to herself

    # Get friends of friends
    potential_friends = set()
    for friend in alice_friends:
        friends_of_friend = db.get_neighbors(friend.id, direction='outgoing', rel_type='FRIENDS_WITH')
        for fof in friends_of_friend:
            if fof.id not in alice_friend_ids:
                potential_friends.add(fof.id)

    print(f"\nRecommended friends for Alice ({len(potential_friends)}):")
    for friend_id in potential_friends:
        friend = db.get_node(friend_id)
        print(f"  - {friend.properties['name']} (@{friend.properties['username']})")

    # =========================================================================
    # Statistics
    # =========================================================================
    print("\n" + "=" * 60)
    print("Network Statistics")
    print("=" * 60)

    total_users = db.get_node_count(label='User')
    total_topics = db.get_node_count(label='Topic')
    total_friendships = db.get_relationship_count(rel_type='FRIENDS_WITH')
    total_follows = db.get_relationship_count(rel_type='FOLLOWS')
    total_interests = db.get_relationship_count(rel_type='INTERESTED_IN')

    print(f"\nUsers: {total_users}")
    print(f"Topics: {total_topics}")
    print(f"Friendships: {total_friendships}")
    print(f"Follow relationships: {total_follows}")
    print(f"Interest connections: {total_interests}")

    all_labels = db.get_all_labels()
    all_rel_types = db.get_all_relationship_types()
    print(f"\nAll labels: {', '.join(all_labels)}")
    print(f"All relationship types: {', '.join(all_rel_types)}")

    # Cleanup
    db.close()
    print("\n" + "=" * 60)
    print("Social network example completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
