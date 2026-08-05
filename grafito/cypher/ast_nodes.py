"""AST (Abstract Syntax Tree) node definitions for Cypher queries."""

from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================================
# Base Classes
# ============================================================================

@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    pass


@dataclass
class Expression(ASTNode):
    """Base class for expressions (used in WHERE clauses)."""
    pass


# ============================================================================
# Variables
# ============================================================================

@dataclass
class Variable(Expression):
    """Variable reference (e.g., n)."""
    name: str


# ============================================================================
# Literal Values
# ============================================================================

@dataclass
class Literal(Expression):
    """Literal value (string, number, boolean, null)."""
    value: Any


@dataclass
class Parameter(Expression):
    """Query parameter reference: $name, resolved from the params passed to execute()."""
    name: str


# ============================================================================
# Property Access
# ============================================================================

@dataclass
class PropertyAccess(Expression):
    """Property access expression: variable.property (e.g., n.age)."""
    variable: str
    property: str


# ============================================================================
# Property Lookup
# ============================================================================

@dataclass
class PropertyLookup(Expression):
    """Property access expression: expr.property (e.g., value.items[0].text)."""
    base_expr: Expression
    property: str


# ============================================================================
# Function Calls
# ============================================================================

@dataclass
class FunctionCall(Expression):
    """Function call expression: COUNT(n), SUM(n.age), etc.

    Examples:
        COUNT(n)
        COUNT(DISTINCT n)
        COUNT(*)
        SUM(n.age)
        AVG(n.salary)
        MIN(n.created_at)
        MAX(n.score)
        STDDEV(n.score)
        PERCENTILECONT(n.score, 0.5)
    """
    function_name: str  # 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COLLECT', 'STDDEV', 'PERCENTILECONT'
    arguments: list[Expression] = field(default_factory=list)
    star: bool = False
    distinct: bool = False


# ============================================================================
# Binary Operations
# ============================================================================

@dataclass
class BinaryOp(Expression):
    """Binary operation: left operator right.

    Operators: =, !=, <, >, <=, >=, AND, OR
    """
    left: Expression
    operator: str
    right: Expression


# ============================================================================
# Unary Operations
# ============================================================================

@dataclass
class UnaryOp(Expression):
    """Unary operation: operator operand.

    Operators: NOT
    """
    operator: str
    operand: Expression


@dataclass
class LabelPredicate(Expression):
    """Label test predicate: `n:Label` (or `n:A:B`) used in a WHERE expression.

    True iff the operand is a node carrying *all* of the given labels.
    """
    operand: Expression
    labels: list[str]


# ============================================================================
# List Expressions
# ============================================================================

@dataclass
class ListLiteral(Expression):
    """List literal: [expr1, expr2, ...]."""
    items: list[Expression] = field(default_factory=list)


@dataclass
class ListComprehension(Expression):
    """List comprehension: [var IN list [WHERE cond] | expr]."""
    variable: str
    list_expr: Expression
    projection: Expression
    where_expr: Optional[Expression] = None


@dataclass
class ListIndex(Expression):
    """List indexing: list[index]."""
    list_expr: Expression
    index_expr: Expression


@dataclass
class ListSlice(Expression):
    """List slicing: list[start..end]."""
    list_expr: Expression
    start_expr: Optional[Expression] = None
    end_expr: Optional[Expression] = None


@dataclass
class ListPredicate(Expression):
    """List predicate: ANY/ALL/NONE/SINGLE (var IN list WHERE cond)."""
    predicate: str
    variable: str
    list_expr: Expression
    where_expr: Optional[Expression] = None


@dataclass
class PatternComprehension(Expression):
    """Pattern comprehension: [pattern [WHERE cond] | expr]."""
    pattern: 'Pattern'
    projection: Expression
    where_expr: Optional[Expression] = None


@dataclass
class MapLiteral(Expression):
    """Map literal: {key: value, ...}."""
    items: dict[str, Expression] = field(default_factory=dict)


@dataclass
class ReduceExpression(Expression):
    """Reduce expression: reduce(acc = init, var IN list | expr)."""
    accumulator: str
    init_expr: Expression
    variable: str
    list_expr: Expression
    expression: Expression


@dataclass
class FunctionCallExpression(Expression):
    """Generic function call expression."""
    name: str
    arguments: list[Expression] = field(default_factory=list)


# ============================================================================
# CASE Expressions
# ============================================================================

@dataclass
class CaseWhen(ASTNode):
    """Single WHEN/THEN pair in a CASE expression."""
    when_expr: Expression
    then_expr: Expression


@dataclass
class CaseExpression(Expression):
    """CASE expression (searched or simple)."""
    base_expr: Optional[Expression] = None
    whens: list[CaseWhen] = field(default_factory=list)
    else_expr: Optional[Expression] = None


# ============================================================================
# Patterns
# ============================================================================

@dataclass
class NodePattern(ASTNode):
    """Node pattern: (variable:Label1:Label2 {prop: value}).

    Examples:
        (n:Person)
        (n:Person {name: 'Alice'})
        (:Person)  # anonymous node
        (n)        # variable only
    """
    variable: Optional[str] = None
    labels: list[str] = field(default_factory=list)
    properties: dict[str, Expression] = field(default_factory=dict)


@dataclass
class RelationshipPattern(ASTNode):
    """Relationship pattern: -[variable:TYPE {props}]-> or <-[...]-.

    Examples:
        -[r:KNOWS]->
        <-[r:KNOWS]-
        -[r:KNOWS {since: 2020}]->
        -[:KNOWS]->  # anonymous relationship
    """
    variable: Optional[str] = None
    rel_type: Optional[str] = None
    properties: dict[str, Expression] = field(default_factory=dict)
    direction: str = 'outgoing'  # 'outgoing', 'incoming', 'both'
    min_hops: int = 1
    max_hops: int = 1


@dataclass
class PatternElement(ASTNode):
    """A single element in a pattern chain.

    Can be:
    - Just a node: PatternElement(node=NodePattern(...))
    - Node + relationship: PatternElement(node=..., relationship=...)
    """
    node: NodePattern
    relationship: Optional[RelationshipPattern] = None


@dataclass
class Pattern(ASTNode):
    """A complete pattern: sequence of nodes and relationships.

    Examples:
        (n:Person)
        (a:Person)-[r:KNOWS]->(b:Person)
        (a)-[r1]->(b)-[r2]->(c)
    """
    variable: Optional[str] = None
    elements: list[PatternElement] = field(default_factory=list)


@dataclass
class PatternFunction(ASTNode):
    """Pattern function wrapper (e.g., shortestPath, allShortestPaths)."""
    name: str
    pattern: Pattern
    variable: Optional[str] = None


# ============================================================================
# Clauses
# ============================================================================

@dataclass
class WhereClause(ASTNode):
    """WHERE clause with a condition expression."""
    condition: Expression


@dataclass
class ReturnItem(ASTNode):
    """Single item in RETURN clause.

    Can be:
    - Variable: n
    - Property access: n.name
    """
    expression: Expression
    alias: Optional[str] = None  # For future: RETURN n.name AS name


@dataclass
class ReturnClause(ASTNode):
    """RETURN clause with list of return items."""
    items: list[ReturnItem] = field(default_factory=list)
    distinct: bool = False


@dataclass
class CreateClause(ASTNode):
    """CREATE clause: CREATE pattern."""
    patterns: list[Pattern] = field(default_factory=list)


@dataclass
class MergeClause(ASTNode):
    """MERGE clause: MERGE pattern [ON CREATE SET ...] [ON MATCH SET ...].

    MERGE finds or creates a pattern:
    - If pattern exists: runs ON MATCH SET (if present)
    - If pattern doesn't exist: creates it and runs ON CREATE SET (if present)
    """
    patterns: list[Pattern] = field(default_factory=list)
    on_create_set: Optional['SetClause'] = None
    on_match_set: Optional['SetClause'] = None


@dataclass
class WithClause(ASTNode):
    """WITH clause: WITH items [WHERE] [ORDER BY] [SKIP] [LIMIT] [RETURN].

    Acts as a pipeline/filter between query parts:
    WITH n, COUNT(n.friends) as friendCount
    WHERE friendCount > 5
    RETURN n.name, friendCount
    """
    items: list[ReturnItem] = field(default_factory=list)
    where_clause: Optional[WhereClause] = None
    return_clause: Optional['ReturnClause'] = None
    order_by_clause: Optional['OrderByClause'] = None
    skip_clause: Optional['SkipClause'] = None
    limit_clause: Optional['LimitClause'] = None
    distinct: bool = False


@dataclass
class UnwindClause(ASTNode):
    """UNWIND clause: UNWIND list_expr AS variable."""
    list_expr: Expression
    variable: str


@dataclass
class MatchClause(ASTNode):
    """MATCH clause: [OPTIONAL] MATCH pattern [WHERE] [DELETE/SET/REMOVE] [RETURN] [ORDER BY] [SKIP] [LIMIT]."""
    patterns: list[Pattern] = field(default_factory=list)
    where_clause: Optional[WhereClause] = None
    delete_clause: Optional['DeleteClause'] = None
    set_clause: Optional['SetClause'] = None
    remove_clause: Optional['RemoveClause'] = None
    return_clause: Optional[ReturnClause] = None
    order_by_clause: Optional['OrderByClause'] = None
    skip_clause: Optional['SkipClause'] = None
    limit_clause: Optional['LimitClause'] = None
    optional: bool = False


@dataclass
class DeleteClause(ASTNode):
    """DELETE clause: DELETE variable1, variable2, ..."""
    variables: list[str] = field(default_factory=list)


@dataclass
class SetItem(ASTNode):
    """Single assignment in SET clause."""
    variable: str
    property: Optional[str]
    value: Expression
    operator: str = "="


@dataclass
class SetClause(ASTNode):
    """SET clause: SET n.prop1 = value1, n += map, n = map."""
    items: list[SetItem] = field(default_factory=list)


@dataclass
class LoadCsvClause(ASTNode):
    """LOAD CSV clause: LOAD CSV [WITH HEADERS] FROM 'url' AS row."""
    source: Expression
    variable: str
    with_headers: bool = False


@dataclass
class RemoveItem(ASTNode):
    """Single item to remove: either a property or a label.

    Examples:
        n.property  - Remove property
        n:Label     - Remove label
    """
    variable: str
    property: Optional[str] = None  # For property removal: n.property
    label: Optional[str] = None      # For label removal: n:Label


@dataclass
class RemoveClause(ASTNode):
    """REMOVE clause: REMOVE n.property, n:Label, ..."""
    items: list[RemoveItem] = field(default_factory=list)


@dataclass
class OrderByItem(ASTNode):
    """Single item in ORDER BY clause: expression [ASC|DESC]."""
    expression: Expression
    ascending: bool = True  # True for ASC, False for DESC


@dataclass
class OrderByClause(ASTNode):
    """ORDER BY clause: ORDER BY item1, item2, ..."""
    items: list[OrderByItem] = field(default_factory=list)


@dataclass
class LimitClause(ASTNode):
    """LIMIT clause: LIMIT n."""
    count: int


@dataclass
class SkipClause(ASTNode):
    """SKIP clause: SKIP n."""
    count: int


@dataclass
class CreateIndexClause(ASTNode):
    """CREATE INDEX clause for property indexes."""
    entity: str  # 'node' or 'relationship'
    label_or_type: str
    property: str
    name: Optional[str] = None
    unique: bool = False
    if_not_exists: bool = False


@dataclass
class DropIndexClause(ASTNode):
    """DROP INDEX clause."""
    name: str
    if_exists: bool = False


@dataclass
class ShowIndexesClause(ASTNode):
    """SHOW INDEXES clause."""
    where_expr: Optional[Expression] = None


@dataclass
class CreateConstraintClause(ASTNode):
    """CREATE CONSTRAINT clause."""
    entity: str  # 'node' or 'relationship'
    label_or_type: str
    property: str
    constraint_type: str  # 'UNIQUE', 'EXISTS', 'TYPE'
    type_name: Optional[str] = None
    name: Optional[str] = None
    if_not_exists: bool = False


@dataclass
class DropConstraintClause(ASTNode):
    """DROP CONSTRAINT clause."""
    name: str
    if_exists: bool = False


@dataclass
class ShowConstraintsClause(ASTNode):
    """SHOW CONSTRAINTS clause."""
    where_expr: Optional[Expression] = None


@dataclass
class ForeachClause(ASTNode):
    """FOREACH clause with update actions."""
    variable: str
    list_expr: Expression
    actions: list[ASTNode] = field(default_factory=list)


# ============================================================================
# Top-Level Query
# ============================================================================

@dataclass
@dataclass
class UnionClause(ASTNode):
    """UNION clause joining queries."""
    query: 'Query'
    all: bool = False


@dataclass
class SubqueryClause(ASTNode):
    """CALL { ... } subquery clause."""
    query: 'Query'


@dataclass
class ProcedureCallClause(ASTNode):
    """CALL procedure clause (e.g., CALL db.vector.search(...)).

    ``yield_items`` holds ``(output_name, alias)`` pairs; without an ``AS``
    clause the alias repeats the output name.
    """
    name: str
    arguments: list[Expression] = field(default_factory=list)
    yield_items: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Query(ASTNode):
    """Top-level query node.

    For simple queries: single clause (CREATE, MATCH, MERGE, WITH, CALL)
    For complex queries with WITH: list of clauses in sequence
    """
    clause: Optional[CreateClause | MatchClause | MergeClause | WithClause | SubqueryClause | ProcedureCallClause
                     | UnwindClause | LoadCsvClause | CreateIndexClause | DropIndexClause | ShowIndexesClause
                     | CreateConstraintClause | DropConstraintClause | ShowConstraintsClause
                     | ForeachClause] = None
    clauses: list[CreateClause | MatchClause | MergeClause | WithClause | SubqueryClause | ProcedureCallClause | UnwindClause | LoadCsvClause | ForeachClause] = field(default_factory=list)
    union_clauses: list[UnionClause] = field(default_factory=list)
