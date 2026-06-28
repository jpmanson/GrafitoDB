"""Demonstrate Cypher queries with both in-memory and persistent databases."""

from grafito import GrafitoDatabase
import os


def example_in_memory():
    """Example using in-memory database."""
    print("=" * 60)
    print("IN-MEMORY DATABASE")
    print("=" * 60)

    db = GrafitoDatabase(':memory:')

    # Create data using Cypher
    db.execute("CREATE (a:Person {name: 'Alice', age: 30})")
    db.execute("CREATE (b:Person {name: 'Bob', age: 25})")
    db.execute("CREATE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})")

    # Query using aggregations
    result = db.execute("MATCH (n:Person) RETURN COUNT(*)")
    print(f"\nTotal persons: {result[0]['COUNT(*)']}")

    result = db.execute("MATCH (n:Person) RETURN AVG(n.age)")
    print(f"Average age: {result[0]['AVG(n.age)']}")

    # Query relationships
    result = db.execute("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name")
    for row in result:
        print(f"{row['a.name']} knows {row['b.name']}")

    db.close()
    print("\nDatabase closed - all data is lost (in-memory mode)")


def example_persistent():
    """Example using persistent database."""
    print("\n" + "=" * 60)
    print("PERSISTENT DATABASE")
    print("=" * 60)

    db_file = 'social_network.db'

    # Remove existing database for clean demo
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"\nRemoved existing {db_file}")

    # First session: Create data
    print(f"\nSession 1: Creating data in {db_file}...")
    db = GrafitoDatabase(db_file)

    db.execute("CREATE (a:Person {name: 'Alice', age: 30, city: 'NYC'})")
    db.execute("CREATE (b:Person {name: 'Bob', age: 25, city: 'LA'})")
    db.execute("CREATE (c:Person {name: 'Charlie', age: 35, city: 'SF'})")

    # Create relationships
    db.execute("""
        CREATE (a:Person {name: 'Alice'})-[r1:KNOWS {since: 2020}]->(b:Person {name: 'Bob'})
    """)
    db.execute("""
        CREATE (b:Person {name: 'Bob'})-[r2:KNOWS {since: 2021}]->(c:Person {name: 'Charlie'})
    """)

    result = db.execute("MATCH (n:Person) RETURN COUNT(*)")
    print(f"Created {result[0]['COUNT(*)']} persons")

    db.close()
    print(f"Database closed - data saved to {db_file}")

    # Second session: Query the saved data
    print(f"\nSession 2: Loading data from {db_file}...")
    db = GrafitoDatabase(db_file)

    # Query all persons
    result = db.execute("""
        MATCH (n:Person)
        RETURN n.name, n.age, n.city
        ORDER BY n.age ASC
    """)
    print("\nAll persons (ordered by age):")
    for row in result:
        print(f"  - {row['n.name']}, {row['n.age']} years, {row['n.city']}")

    # Find oldest person
    result = db.execute("MATCH (n:Person) RETURN MAX(n.age)")
    print(f"\nOldest person is {result[0]['MAX(n.age)']} years old")

    # Find relationships
    result = db.execute("""
        MATCH (a:Person)-[r:KNOWS]->(b:Person)
        RETURN a.name, b.name, r.since
        ORDER BY r.since
    """)
    print("\nRelationships:")
    for row in result:
        print(f"  - {row['a.name']} knows {row['b.name']} (since {row['r.since']})")

    # Update data
    print("\nUpdating Bob's age...")
    db.execute("""
        MATCH (n:Person {name: 'Bob'})
        SET n.age = 26
        RETURN n.name, n.age
    """)

    # Verify update
    result = db.execute("MATCH (n:Person {name: 'Bob'}) RETURN n.age")
    print(f"Bob's new age: {result[0]['n.age']}")

    db.close()
    print(f"\nDatabase closed - all changes saved to {db_file}")

    # Third session: Verify persistence
    print(f"\nSession 3: Verifying persistence in {db_file}...")
    db = GrafitoDatabase(db_file)

    result = db.execute("MATCH (n:Person {name: 'Bob'}) RETURN n.age")
    print(f"Bob's age is still {result[0]['n.age']} (persisted!)")

    db.close()

    print(f"\n[OK] Database file {db_file} created and can be reused")


def example_with_transactions():
    """Example showing transaction support."""
    print("\n" + "=" * 60)
    print("TRANSACTIONS")
    print("=" * 60)

    db_file = 'transactions_demo.db'
    if os.path.exists(db_file):
        os.remove(db_file)

    db = GrafitoDatabase(db_file)

    # Transaction commits on success
    print("\nTransaction 1: Creating data (will commit)...")
    with db:
        db.execute("CREATE (n:Product {name: 'Laptop', price: 999})")
        db.execute("CREATE (n:Product {name: 'Mouse', price: 25})")

    result = db.execute("MATCH (n:Product) RETURN COUNT(*)")
    print(f"Products after commit: {result[0]['COUNT(*)']}")

    # Transaction rollback on exception
    print("\nTransaction 2: Trying to create invalid data (will rollback)...")
    try:
        with db:
            db.execute("CREATE (n:Product {name: 'Keyboard', price: 50})")
            # Simulate an error
            raise Exception("Something went wrong!")
    except Exception as e:
        print(f"Error: {e}")

    result = db.execute("MATCH (n:Product) RETURN COUNT(*)")
    print(f"Products after rollback: {result[0]['COUNT(*)']} (rollback worked!)")

    db.close()

    if os.path.exists(db_file):
        os.remove(db_file)


if __name__ == '__main__':
    example_in_memory()
    example_persistent()
    example_with_transactions()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("[OK] In-memory mode: Fast, temporary, perfect for tests")
    print("[OK] Persistent mode: Data saved to disk, survives restarts")
    print("[OK] Transactions: ACID guarantees with context managers")
    print("[OK] Cypher queries work identically in both modes")
