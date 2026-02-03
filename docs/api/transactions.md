# Transactions

GrafitoDB provides full ACID transaction support through SQLite.

## What are Transactions?

A transaction groups multiple operations into a single atomic unit. Either all operations succeed (commit) or none do (rollback).

```
BEGIN TRANSACTION
  Operation 1 ✓
  Operation 2 ✓
  Operation 3 ✗ (fails)
ROLLBACK  ← All undone

BEGIN TRANSACTION
  Operation 1 ✓
  Operation 2 ✓
  Operation 3 ✓
COMMIT    ← All persisted
```

## Using Context Managers (Recommended)

The simplest and safest way to use transactions:

```python
from grafito import GrafitoDatabase

db = GrafitoDatabase(':memory:')

# Use as context manager
with db:
    alice = db.create_node(labels=['Person'], properties={'name': 'Alice'})
    bob = db.create_node(labels=['Person'], properties={'name': 'Bob'})
    db.create_relationship(alice.id, bob.id, 'KNOWS')

# Automatically commits if no exception
# Automatically rolls back if any exception occurs
```

### Automatic Rollback Example

```python
def create_user_with_validation(db, username, email):
    with db:
        # Step 1: Create user node
        user = db.create_node(
            labels=['User'],
            properties={'username': username, 'email': email}
        )
        
        # Step 2: This will fail (invalid email)
        if '@' not in email:
            raise ValueError("Invalid email")
        
        # Step 3: Create profile node
        profile = db.create_node(
            labels=['Profile'],
            properties={'user_id': user.id}
        )
        
        db.create_relationship(user.id, profile.id, 'HAS_PROFILE')
    
    # If email is invalid:
    # - user node is NOT created
    # - profile node is NOT created
    # - relationship is NOT created
```

## Manual Transaction Control

For more control, use explicit methods:

```python
db.begin_transaction()

try:
    # Perform operations
    node1 = db.create_node(labels=['A'])
    node2 = db.create_node(labels=['B'])
    db.create_relationship(node1.id, node2.id, 'LINKS')
    
    # Verify before committing
    if is_valid_structure(node1, node2):
        db.commit()
        print("Transaction committed")
    else:
        db.rollback()
        print("Transaction rolled back (validation failed)")
        
except Exception as e:
    db.rollback()
    print(f"Transaction rolled back (error: {e})")
    raise
```

## Nested Operations

Transactions can span multiple method calls:

```python
def create_organization(db, org_data):
    """Create org structure in one transaction."""
    with db:
        # Create organization
        org = db.create_node(
            labels=['Organization'],
            properties={'name': org_data['name']}
        )
        
        # Create departments
        for dept_data in org_data['departments']:
            create_department(db, org.id, dept_data)
        
        return org

def create_department(db, org_id, dept_data):
    """Helper function - runs in parent's transaction."""
    dept = db.create_node(
        labels=['Department'],
        properties={'name': dept_data['name']}
    )
    db.create_relationship(org_id, dept.id, 'HAS_DEPARTMENT')
    
    # Add employees
    for emp in dept_data['employees']:
        employee = db.create_node(
            labels=['Employee'],
            properties=emp
        )
        db.create_relationship(employee.id, dept.id, 'WORKS_IN')

# Usage - all or nothing:
with db:
    create_organization(db, {
        'name': 'TechCorp',
        'departments': [
            {
                'name': 'Engineering',
                'employees': [
                    {'name': 'Alice', 'role': 'Developer'},
                    {'name': 'Bob', 'role': 'Designer'}
                ]
            }
        ]
    })
```

## Savepoints (Nested Transactions)

SQLite supports savepoints for partial rollback:

```python
with db:
    # Main transaction
    user = db.create_node(labels=['User'], properties={'name': 'Alice'})
    
    try:
        # This could fail independently
        with db:
            profile = db.create_node(labels=['Profile'])
            db.create_relationship(user.id, profile.id, 'HAS_PROFILE')
    except Exception:
        # Profile creation failed, but user is kept
        pass
    
    # This always runs if user was created
    settings = db.create_node(labels=['Settings'])
    db.create_relationship(user.id, settings.id, 'HAS_SETTINGS')
```

## Best Practices

### 1. Keep Transactions Short

```python
# Good: Short, focused transaction
with db:
    user = db.create_node(labels=['User'], properties=data)
    db.create_relationship(user.id, org.id, 'MEMBER_OF')

# Process outside transaction (no DB locks)
send_welcome_email(user.properties['email'])
```

### 2. Handle Errors Gracefully

```python
def safe_create_user(db, user_data):
    try:
        with db:
            # Check for existing user
            existing = db.match_nodes(
                labels=['User'],
                properties={'email': user_data['email']}
            )
            if existing:
                raise ValueError("User already exists")
            
            return db.create_node(labels=['User'], properties=user_data)
            
    except ValueError as e:
        # Expected error - user exists
        logger.info(f"User creation skipped: {e}")
        return None
    except Exception as e:
        # Unexpected error
        logger.error(f"Database error: {e}")
        raise
```

### 3. Validate Before Transaction

```python
def create_product_with_validation(db, product_data):
    # Validation outside transaction (no locks)
    if not product_data.get('name'):
        raise ValueError("Product name required")
    if product_data.get('price', 0) <= 0:
        raise ValueError("Invalid price")
    
    # Now do the transaction
    with db:
        product = db.create_node(
            labels=['Product'],
            properties=product_data
        )
        
        # Create inventory record
        inventory = db.create_node(
            labels=['Inventory'],
            properties={'product_id': product.id, 'quantity': 0}
        )
        db.create_relationship(product.id, inventory.id, 'HAS_INVENTORY')
        
        return product
```

### 4. Read-Only Operations

Read operations don't need explicit transactions (SQLite handles consistency):

```python
# No transaction needed for reads
user = db.get_node(user_id)
users = db.match_nodes(labels=['User'])

# Transaction needed for writes
with db:
    db.update_node_properties(user_id, {'last_seen': now()})
```

## Common Patterns

### Bulk Insert

```python
def bulk_create_nodes(db, items):
    """Efficiently create many nodes."""
    with db:
        created = []
        for item in items:
            node = db.create_node(
                labels=item['labels'],
                properties=item['properties']
            )
            created.append(node)
        return created
```

### Cascade Delete

```python
def delete_user_with_data(db, user_id):
    """Delete user and all related data."""
    with db:
        # Find related nodes (assuming specific structure)
        results = db.execute("""
            MATCH (u:User {id: $user_id})-[:HAS_PROFILE|HAS_SETTINGS]->(related)
            RETURN related.id as related_id
        """, {'user_id': user_id})
        
        # Delete related nodes first
        for row in results:
            db.delete_node(row['related_id'])
        
        # Delete user (cascades relationships)
        db.delete_node(user_id)
```

### Conditional Updates

```python
def transfer_funds(db, from_id, to_id, amount):
    """Transfer between accounts atomically."""
    with db:
        from_acc = db.get_node(from_id)
        to_acc = db.get_node(to_id)
        
        if not from_acc or not to_acc:
            raise ValueError("Account not found")
        
        current_balance = from_acc.properties.get('balance', 0)
        if current_balance < amount:
            raise ValueError("Insufficient funds")
        
        # Update both accounts atomically
        db.update_node_properties(
            from_id,
            {'balance': current_balance - amount}
        )
        db.update_node_properties(
            to_id,
            {'balance': to_acc.properties.get('balance', 0) + amount}
        )
```

## Error Types

| Exception | When Raised |
|-----------|-------------|
| `TransactionError` | Invalid transaction state |
| `Rollback` | Explicit rollback requested |
| `ConstraintError` | Unique constraint violation |
| `NodeNotFoundError` | Referenced node doesn't exist |
