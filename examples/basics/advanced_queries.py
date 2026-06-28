"""Advanced query examples for Grafito graph database.

This script demonstrates the advanced search and filtering capabilities
introduced in the enhanced API, including:
- Comparison operators (>, <, >=, <=, !=, BETWEEN)
- String pattern matching (CONTAINS, STARTS_WITH, ENDS_WITH, REGEX)
- OR logic for labels and properties
- Ordering and pagination

All features are backward compatible with existing code.
"""

from grafito import (
    GrafitoDatabase,
    PropertyFilter,
    PropertyFilterGroup,
    LabelFilter,
    SortOrder,
)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)


def print_nodes(nodes, title="Results"):
    """Print a list of nodes."""
    print(f"\n{title}: {len(nodes)} nodes")
    for node in nodes:
        print(f"  - Node {node.id}: {node.labels} {node.properties}")


def example_comparison_operators():
    """Demonstrate comparison operators."""
    print_section("1. Comparison Operators")

    db = GrafitoDatabase()

    # Create sample data
    print("\nCreating sample persons...")
    persons = [
        db.create_node(labels=['Person'], properties={'name': 'Alice', 'age': 25, 'salary': 50000}),
        db.create_node(labels=['Person'], properties={'name': 'Bob', 'age': 30, 'salary': 65000}),
        db.create_node(labels=['Person'], properties={'name': 'Charlie', 'age': 35, 'salary': 80000}),
        db.create_node(labels=['Person'], properties={'name': 'Diana', 'age': 28, 'salary': 55000}),
        db.create_node(labels=['Person'], properties={'name': 'Eve', 'age': 40, 'salary': 95000}),
    ]

    # Greater than
    print("\n[1.1] Age > 30:")
    results = db.match_nodes(
        labels=['Person'],
        properties={'age': PropertyFilter.gt(30)}
    )
    print_nodes(results)

    # Less than or equal
    print("\n[1.2] Age <= 28:")
    results = db.match_nodes(
        labels=['Person'],
        properties={'age': PropertyFilter.lte(28)}
    )
    print_nodes(results)

    # Between (salary range)
    print("\n[1.3] Salary between $50,000 and $70,000:")
    results = db.match_nodes(
        labels=['Person'],
        properties={'salary': PropertyFilter.between(50000, 70000)}
    )
    print_nodes(results)

    # Not equal
    print("\n[1.4] Name != 'Alice':")
    results = db.match_nodes(
        labels=['Person'],
        properties={'name': PropertyFilter.ne('Alice')}
    )
    print_nodes(results)

    db.close()


def example_string_matching():
    """Demonstrate string pattern matching."""
    print_section("2. String Pattern Matching")

    db = GrafitoDatabase()

    # Create sample data
    print("\nCreating sample users...")
    users = [
        db.create_node(labels=['User'], properties={'name': 'Alice Johnson', 'email': 'alice@example.com'}),
        db.create_node(labels=['User'], properties={'name': 'Bob Smith', 'email': 'bob@gmail.com'}),
        db.create_node(labels=['User'], properties={'name': 'Charlie Brown', 'email': 'charlie@example.com'}),
        db.create_node(labels=['User'], properties={'name': 'Diana Prince', 'email': 'diana@yahoo.com'}),
        db.create_node(labels=['User'], properties={'name': 'Eve Anderson', 'email': 'eve@gmail.com'}),
    ]

    # Contains (case-insensitive)
    print("\n[2.1] Name contains 'son' (case-insensitive):")
    results = db.match_nodes(
        labels=['User'],
        properties={'name': PropertyFilter.contains('son', case_sensitive=False)}
    )
    print_nodes(results)

    # Starts with
    print("\n[2.2] Name starts with 'A':")
    results = db.match_nodes(
        labels=['User'],
        properties={'name': PropertyFilter.starts_with('A')}
    )
    print_nodes(results)

    # Ends with
    print("\n[2.3] Email ends with '@gmail.com':")
    results = db.match_nodes(
        labels=['User'],
        properties={'email': PropertyFilter.ends_with('@gmail.com')}
    )
    print_nodes(results)

    # Regex pattern
    print("\n[2.4] Email matches pattern '.*@example.com':")
    results = db.match_nodes(
        labels=['User'],
        properties={'email': PropertyFilter.regex(r'.*@example\.com')}
    )
    print_nodes(results)

    db.close()


def example_or_logic():
    """Demonstrate OR logic for labels and properties."""
    print_section("3. OR Logic")

    db = GrafitoDatabase()

    # Create sample data
    print("\nCreating sample entities...")
    alice = db.create_node(labels=['Person', 'Employee'], properties={'name': 'Alice', 'city': 'NYC'})
    tech_corp = db.create_node(labels=['Company'], properties={'name': 'TechCorp', 'city': 'SF'})
    bob = db.create_node(labels=['Person'], properties={'name': 'Bob', 'city': 'LA'})
    startup_inc = db.create_node(labels=['Company'], properties={'name': 'StartupInc', 'city': 'NYC'})
    charlie = db.create_node(labels=['Person'], properties={'name': 'Charlie', 'city': 'NYC'})

    # Label OR
    print("\n[3.1] Entities with label Person OR Company:")
    results = db.match_nodes(
        labels=LabelFilter.any(['Person', 'Company'])
    )
    print_nodes(results)

    # Property OR (city is NYC OR LA)
    print("\n[3.2] Persons in NYC OR LA:")
    results = db.match_nodes(
        labels=['Person'],
        properties=PropertyFilterGroup.or_(
            {'city': 'NYC'},
            {'city': 'LA'}
        )
    )
    print_nodes(results)

    # Complex: (city=NYC AND label=Person) OR (city=SF AND label=Company)
    print("\n[3.3] Persons in NYC OR Companies in SF:")
    nyc_persons = db.match_nodes(labels=['Person'], properties={'city': 'NYC'})
    sf_companies = db.match_nodes(labels=['Company'], properties={'city': 'SF'})
    print(f"\nNYC Persons: {len(nyc_persons)}, SF Companies: {len(sf_companies)}")
    print("Combined approach using label OR with filters:")
    results = db.match_nodes(
        labels=LabelFilter.any(['Person', 'Company'])
    )
    filtered = [n for n in results if
                (n.properties.get('city') == 'NYC' and 'Person' in n.labels) or
                (n.properties.get('city') == 'SF' and 'Company' in n.labels)]
    print_nodes(filtered, "Filtered results")

    db.close()


def example_ordering():
    """Demonstrate ordering capabilities."""
    print_section("4. Ordering")

    db = GrafitoDatabase()

    # Create sample data
    print("\nCreating sample products...")
    products = [
        db.create_node(labels=['Product'], properties={'name': 'Laptop', 'price': 1200, 'rating': 4.5}),
        db.create_node(labels=['Product'], properties={'name': 'Mouse', 'price': 25, 'rating': 4.8}),
        db.create_node(labels=['Product'], properties={'name': 'Keyboard', 'price': 75, 'rating': 4.6}),
        db.create_node(labels=['Product'], properties={'name': 'Monitor', 'price': 300, 'rating': 4.4}),
        db.create_node(labels=['Product'], properties={'name': 'Headphones', 'price': 150, 'rating': 4.7}),
    ]

    # Sort by single property (ascending)
    print("\n[4.1] Products sorted by price (low to high):")
    results = db.match_nodes(
        labels=['Product'],
        order_by='price',
        ascending=True
    )
    print_nodes(results)

    # Sort by single property (descending)
    print("\n[4.2] Products sorted by rating (high to low):")
    results = db.match_nodes(
        labels=['Product'],
        order_by='rating',
        ascending=False
    )
    print_nodes(results)

    # Sort by multiple properties
    print("\n[4.3] Products sorted by rating DESC, then price ASC:")
    results = db.match_nodes(
        labels=['Product'],
        order_by=[
            SortOrder('rating', ascending=False),
            SortOrder('price', ascending=True)
        ]
    )
    print_nodes(results)

    db.close()


def example_pagination():
    """Demonstrate pagination."""
    print_section("5. Pagination")

    db = GrafitoDatabase()

    # Create sample data
    print("\nCreating 20 sample items...")
    for i in range(1, 21):
        db.create_node(labels=['Item'], properties={'number': i, 'name': f'Item {i}'})

    # Page 1 (items 1-5)
    print("\n[5.1] Page 1 (limit=5, offset=0):")
    results = db.match_nodes(
        labels=['Item'],
        order_by='number',
        limit=5,
        offset=0
    )
    print_nodes(results)

    # Page 2 (items 6-10)
    print("\n[5.2] Page 2 (limit=5, offset=5):")
    results = db.match_nodes(
        labels=['Item'],
        order_by='number',
        limit=5,
        offset=5
    )
    print_nodes(results)

    # Page 3 (items 11-15)
    print("\n[5.3] Page 3 (limit=5, offset=10):")
    results = db.match_nodes(
        labels=['Item'],
        order_by='number',
        limit=5,
        offset=10
    )
    print_nodes(results)

    db.close()


def example_combined_features():
    """Demonstrate combining multiple advanced features."""
    print_section("6. Combined Features (E-commerce Example)")

    db = GrafitoDatabase()

    # Create sample products
    print("\nCreating sample e-commerce products...")
    products = [
        db.create_node(labels=['Product', 'Electronics'], properties={
            'name': 'Laptop Pro', 'price': 1500, 'rating': 4.8, 'in_stock': True
        }),
        db.create_node(labels=['Product', 'Electronics'], properties={
            'name': 'Tablet Max', 'price': 800, 'rating': 4.5, 'in_stock': True
        }),
        db.create_node(labels=['Product', 'Books'], properties={
            'name': 'Python Guide', 'price': 45, 'rating': 4.9, 'in_stock': True
        }),
        db.create_node(labels=['Product', 'Electronics'], properties={
            'name': 'Phone Ultra', 'price': 1200, 'rating': 4.7, 'in_stock': False
        }),
        db.create_node(labels=['Product', 'Books'], properties={
            'name': 'Graph Theory', 'price': 55, 'rating': 4.6, 'in_stock': True
        }),
        db.create_node(labels=['Product', 'Electronics'], properties={
            'name': 'Smart Watch', 'price': 400, 'rating': 4.3, 'in_stock': True
        }),
    ]

    # Complex query: Electronics OR Books, price 40-1000, rating >= 4.5,
    # in stock, sorted by rating DESC, limit 3
    print("\n[6.1] Find top 3 products:")
    print("  - Category: Electronics OR Books")
    print("  - Price: $40 - $1000")
    print("  - Rating: >= 4.5")
    print("  - In stock: Yes")
    print("  - Sort by: Rating (high to low)")
    print("  - Limit: 3")

    results = db.match_nodes(
        labels=LabelFilter.any(['Electronics', 'Books']),
        properties={
            'price': PropertyFilter.between(40, 1000),
            'rating': PropertyFilter.gte(4.5),
            'in_stock': True
        },
        order_by='rating',
        ascending=False,
        limit=3
    )
    print_nodes(results)

    db.close()


def example_relationships():
    """Demonstrate advanced relationship filtering."""
    print_section("7. Advanced Relationship Filtering")

    db = GrafitoDatabase()

    # Create sample data
    print("\nCreating sample social network...")
    alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
    bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
    charlie = db.create_node(labels=['Person'], properties={'name': 'Charlie'})
    diana = db.create_node(labels=['Person'], properties={'name': 'Diana'})

    # Create friendships with timestamps
    db.create_relationship(alice.id, bob.id, 'FRIENDS_WITH', {'since': 2018, 'closeness': 9})
    db.create_relationship(alice.id, charlie.id, 'FRIENDS_WITH', {'since': 2020, 'closeness': 7})
    db.create_relationship(bob.id, diana.id, 'FRIENDS_WITH', {'since': 2019, 'closeness': 8})
    db.create_relationship(charlie.id, diana.id, 'FRIENDS_WITH', {'since': 2021, 'closeness': 6})

    # Find friendships since 2019 or later
    print("\n[7.1] Friendships since 2019 or later:")
    results = db.match_relationships(
        rel_type='FRIENDS_WITH',
        properties={'since': PropertyFilter.gte(2019)}
    )
    print(f"\nResults: {len(results)} relationships")
    for rel in results:
        source = db.get_node(rel.source_id)
        target = db.get_node(rel.target_id)
        print(f"  - {source.properties['name']} -> {target.properties['name']}: {rel.properties}")

    # Find close friendships (closeness > 7), ordered by closeness
    print("\n[7.2] Close friendships (closeness > 7), ordered by closeness:")
    results = db.match_relationships(
        rel_type='FRIENDS_WITH',
        properties={'closeness': PropertyFilter.gt(7)},
        order_by='closeness',
        ascending=False
    )
    print(f"\nResults: {len(results)} relationships")
    for rel in results:
        source = db.get_node(rel.source_id)
        target = db.get_node(rel.target_id)
        print(f"  - {source.properties['name']} -> {target.properties['name']}: {rel.properties}")

    db.close()


def example_backward_compatibility():
    """Demonstrate that old code still works (backward compatibility)."""
    print_section("8. Backward Compatibility")

    db = GrafitoDatabase()

    print("\nAll existing API calls work unchanged:")

    # Old style: exact match
    print("\n[8.1] Creating nodes with old API:")
    alice = db.create_node(labels=['Person'], properties={'name': 'Alice', 'age': 30})
    bob = db.create_node(labels=['Person', 'Employee'], properties={'name': 'Bob', 'age': 25})
    print(f"  - Created node {alice.id}: {alice.properties}")
    print(f"  - Created node {bob.id}: {bob.properties}")

    # Old style: match by label
    print("\n[8.2] Match by label (old API):")
    results = db.match_nodes(labels=['Person'])
    print_nodes(results)

    # Old style: match by label AND properties
    print("\n[8.3] Match by label AND properties (old API):")
    results = db.match_nodes(labels=['Person', 'Employee'], properties={'name': 'Bob'})
    print_nodes(results)

    # Old style: match relationships
    print("\n[8.4] Creating and matching relationships (old API):")
    company = db.create_node(labels=['Company'], properties={'name': 'TechCorp'})
    rel = db.create_relationship(bob.id, company.id, 'WORKS_AT', {'since': 2020})
    print(f"  - Created relationship {rel.id}")

    results = db.match_relationships(source_id=bob.id, rel_type='WORKS_AT')
    print(f"  - Found {len(results)} relationships")

    print("\n100% backward compatible!")

    db.close()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("GRAFITO ADVANCED QUERY API - EXAMPLES")
    print("="*60)

    example_comparison_operators()
    example_string_matching()
    example_or_logic()
    example_ordering()
    example_pagination()
    example_combined_features()
    example_relationships()
    example_backward_compatibility()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
