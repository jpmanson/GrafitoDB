# Example: Northwind

This example loads the classic Northwind dataset and runs verification checks.

Source: `examples/northwind.py` and `examples/northwind.cypher`.

## What It Loads

Labels:

- `Product` (77)
- `Category` (8)
- `Supplier` (29)
- `Customer` (91)
- `Order` (830)

Relationship types:

- `PART_OF` (Product -> Category)
- `SUPPLIES` (Supplier -> Product)
- `PURCHASED` (Customer -> Order)
- `ORDERS` (Order -> Product)

## Run the Loader

```bash
python examples/northwind.py
```

By default, the loader uses an in-memory database. You can also use a file:

```bash
python examples/northwind.py --db northwind.db --clean
```

## What It Checks

The loader validates:

- Node counts for each label
- Relationship counts for each type
- Field types (e.g., `unitPrice` is a float, `discontinued` is a boolean)

## Inspect the Data

You can open a database and run Cypher queries:

```python
from grafito import GrafitoDatabase

db = GrafitoDatabase("northwind.db")

results = db.execute("MATCH (p:Product) RETURN p.productName AS name LIMIT 5")
print(results)
```
