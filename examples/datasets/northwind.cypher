// ============================================
// 1. CREATE UNIQUENESS CONSTRAINTS
// ============================================
// Ensure that IDs for each node type are unique in the database

CREATE CONSTRAINT Product_productID IF NOT EXISTS FOR (p:Product) REQUIRE (p.productID) IS UNIQUE;
CREATE CONSTRAINT Category_categoryID IF NOT EXISTS FOR (c:Category) REQUIRE (c.categoryID) IS UNIQUE;
CREATE CONSTRAINT Supplier_supplierID IF NOT EXISTS FOR (s:Supplier) REQUIRE (s.supplierID) IS UNIQUE;
CREATE CONSTRAINT Customer_customerID IF NOT EXISTS FOR (c:Customer) REQUIRE (c.customerID) IS UNIQUE;
CREATE CONSTRAINT Order_orderID IF NOT EXISTS FOR (o:Order) REQUIRE (o.orderID) IS UNIQUE;

// ============================================
// 2. LOAD PRODUCTS
// ============================================
// Import products from CSV and convert numeric and boolean data types

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/products.csv" AS row
MERGE (n:Product {productID:row.productID})
SET n += row,
n.unitPrice = toFloat(row.unitPrice),
n.unitsInStock = toInteger(row.unitsInStock), n.unitsOnOrder = toInteger(row.unitsOnOrder),
n.reorderLevel = toInteger(row.reorderLevel), n.discontinued = (row.discontinued <> "0");

// ============================================
// 3. LOAD CATEGORIES
// ============================================
// Import product categories from CSV

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/categories.csv" AS row
MERGE (n:Category {categoryID:row.categoryID})
SET n += row;

// ============================================
// 4. LOAD SUPPLIERS
// ============================================
// Import suppliers from CSV

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/suppliers.csv" AS row
MERGE (n:Supplier {supplierID:row.supplierID})
SET n += row;

// ============================================
// 5. CREATE RELATIONSHIPS: PRODUCT -> CATEGORY
// ============================================
// Connect each product with its corresponding category using the PART_OF relationship

MATCH (p:Product),(c:Category)
WHERE p.categoryID = c.categoryID
MERGE (p)-[:PART_OF]->(c);

// ============================================
// 6. CREATE RELATIONSHIPS: SUPPLIER -> PRODUCT
// ============================================
// Connect each supplier with the products they supply using the SUPPLIES relationship

MATCH (p:Product),(s:Supplier)
WHERE p.supplierID = s.supplierID
MERGE (s)-[:SUPPLIES]->(p);

// ============================================
// 7. LOAD CUSTOMERS
// ============================================
// Import customers from CSV

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/customers.csv" AS row
MERGE (n:Customer {customerID:row.customerID})
SET n += row;

// ============================================
// 8. LOAD ORDERS
// ============================================
// Import purchase orders from CSV

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/orders.csv" AS row
MERGE (n:Order {orderID:row.orderID})
SET n += row;

// ============================================
// 9. CREATE RELATIONSHIPS: CUSTOMER -> ORDER
// ============================================
// Connect each customer with their purchase orders using the PURCHASED relationship

MATCH (c:Customer),(o:Order)
WHERE c.customerID = o.customerID
MERGE (c)-[:PURCHASED]->(o);

// ============================================
// 10. LOAD ORDER DETAILS
// ============================================
// Create relationships between orders and products with details like quantity
// Each ORDERS relationship represents a specific product within an order

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/order-details.csv" AS row
MATCH (p:Product), (o:Order)
WHERE p.productID = row.productID AND o.orderID = row.orderID
MERGE (o)-[details:ORDERS]->(p)
SET details = row,
details.quantity = toInteger(row.quantity);