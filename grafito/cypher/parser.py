"""Parser for Cypher queries - converts tokens to AST."""

from typing import Optional
from .tokens import Token, TokenType, KEYWORDS
from .ast_nodes import (
    Query, CreateClause, MergeClause, MatchClause, WithClause, WhereClause, ReturnClause,
    DeleteClause, SetClause, SetItem, RemoveClause, RemoveItem,
    OrderByClause, OrderByItem, LimitClause, SkipClause,
    Pattern, PatternElement, NodePattern, RelationshipPattern, PatternFunction,
    ReturnItem, Expression, Literal, PropertyAccess, PropertyLookup, FunctionCall, BinaryOp, UnaryOp, LabelPredicate,
    CaseExpression, CaseWhen, UnionClause, SubqueryClause, ProcedureCallClause, ListLiteral, ListComprehension,
    ListIndex, ListSlice, ListPredicate, FunctionCallExpression, Variable, MapLiteral,
    ReduceExpression, PatternComprehension, UnwindClause, LoadCsvClause,
    CreateIndexClause, DropIndexClause, ShowIndexesClause,
    CreateConstraintClause, DropConstraintClause, ShowConstraintsClause,
    ForeachClause
)
from .exceptions import CypherSyntaxError


class Parser:
    """Recursive descent parser for Cypher queries."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self._symbolic_name_tokens = set(KEYWORDS.values()) - {TokenType.BOOLEAN, TokenType.NULL}
        self._name_tokens = self._symbolic_name_tokens | {TokenType.IDENTIFIER}

    def current_token(self) -> Token:
        """Get current token."""
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[self.pos]

    def peek_token(self, offset: int = 1) -> Token:
        """Peek ahead at a token."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[pos]

    def advance(self) -> Token:
        """Move to next token and return current."""
        token = self.current_token()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type and advance, or raise error."""
        token = self.current_token()
        if token.type != token_type:
            raise CypherSyntaxError(
                f"Expected {token_type.name}, got {token.type.name}",
                token.line,
                token.column
            )
        return self.advance()

    def parse(self) -> Query:
        """Parse a complete query.

        Can be a simple single-clause query or multi-clause query with WITH.

        Examples:
            MATCH (n) RETURN n              -- Single clause
            MATCH (n) WITH n MATCH (m) RETURN n, m  -- Multi-clause with WITH
        """
        base_query = self._parse_single_query()
        union_clauses = []

        while self.current_token().type == TokenType.UNION:
            self.advance()
            union_all = False
            if self.current_token().type == TokenType.ALL:
                union_all = True
                self.advance()
            union_query = self._parse_single_query()
            union_clauses.append(UnionClause(query=union_query, all=union_all))

        if union_clauses:
            base_query.union_clauses = union_clauses
        return base_query

    def _parse_single_query(self) -> Query:
        """Parse a single query (no UNION)."""
        clauses = []

        # A query may consist solely of a RETURN (e.g. `RETURN 1 + 1 AS x`).
        if self.current_token().type == TokenType.RETURN:
            return Query(clause=self._parse_return())

        first_clause = self._parse_next_clause(allow_with_start=True)
        clauses.append(first_clause)

        # Parse remaining clauses until UNION or EOF
        while self.current_token().type not in (TokenType.UNION, TokenType.EOF):
            if self.current_token().type == TokenType.RETURN:
                clauses.append(self._parse_return())
                break
            clauses.append(self._parse_next_clause(allow_with_start=True))

        if len(clauses) == 1:
            return Query(clause=clauses[0])
        return Query(clauses=clauses)

    def _parse_next_clause(self, allow_with_start: bool) -> CreateClause | MergeClause | MatchClause | WithClause | SubqueryClause | ProcedureCallClause | UnwindClause | LoadCsvClause | CreateIndexClause | DropIndexClause | ShowIndexesClause | CreateConstraintClause | DropConstraintClause | ShowConstraintsClause | ForeachClause | SetClause:
        """Parse the next top-level clause."""
        token = self.current_token()
        if token.type == TokenType.CREATE:
            if self.peek_token().type in (TokenType.INDEX, TokenType.UNIQUE):
                return self._parse_create_index()
            if self.peek_token().type == TokenType.CONSTRAINT:
                return self._parse_create_constraint()
            return self._parse_create()
        if token.type == TokenType.MERGE:
            return self._parse_merge()
        if token.type == TokenType.FOREACH:
            return self._parse_foreach()
        if token.type == TokenType.DROP:
            if self.peek_token().type == TokenType.INDEX:
                return self._parse_drop_index()
            if self.peek_token().type == TokenType.CONSTRAINT:
                return self._parse_drop_constraint()
            raise CypherSyntaxError(
                f"Expected INDEX after DROP, got {self.peek_token().type.name}",
                self.peek_token().line,
                self.peek_token().column
            )
        if token.type == TokenType.SHOW:
            if self.peek_token().type == TokenType.CONSTRAINTS:
                return self._parse_show_constraints()
            return self._parse_show_indexes()
        if token.type == TokenType.CALL:
            return self._parse_call()
        if token.type == TokenType.UNWIND:
            return self._parse_unwind()
        if token.type == TokenType.LOAD:
            return self._parse_load_csv()
        if token.type == TokenType.SET:
            return self._parse_set()
        if token.type == TokenType.WITH:
            if not allow_with_start:
                raise CypherSyntaxError(
                    f"Unexpected WITH clause at this position",
                    token.line,
                    token.column
                )
            return self._parse_with()
        if token.type == TokenType.OPTIONAL:
            self.advance()
            if self.current_token().type != TokenType.MATCH:
                raise CypherSyntaxError(
                    f"Expected MATCH after OPTIONAL, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            return self._parse_match(optional=True)
        if token.type == TokenType.MATCH:
            return self._parse_match(optional=False)

        raise CypherSyntaxError(
            f"Expected CREATE, MERGE, DROP, SHOW, CALL, OPTIONAL, MATCH, UNWIND, LOAD, SET, or WITH, got {token.type.name}",
            token.line,
            token.column
        )

    def _parse_foreach(self) -> ForeachClause:
        """Parse FOREACH (var IN list | action...)."""
        self.expect(TokenType.FOREACH)
        self.expect(TokenType.LPAREN)

        if self.current_token().type != TokenType.IDENTIFIER:
            raise CypherSyntaxError(
                f"Expected identifier in FOREACH, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        variable = self.current_token().value
        self.advance()
        self.expect(TokenType.IN)

        list_expr = self._parse_expression()
        self.expect(TokenType.PIPE)

        actions = []
        while self.current_token().type != TokenType.RPAREN:
            if self.current_token().type == TokenType.CREATE:
                actions.append(self._parse_create())
            elif self.current_token().type == TokenType.MERGE:
                actions.append(self._parse_merge())
            elif self.current_token().type == TokenType.SET:
                actions.append(self._parse_set())
            elif self.current_token().type == TokenType.REMOVE:
                actions.append(self._parse_remove())
            elif self.current_token().type == TokenType.DELETE:
                actions.append(self._parse_delete())
            else:
                raise CypherSyntaxError(
                    f"Expected CREATE, MERGE, SET, REMOVE, or DELETE in FOREACH, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            if self.current_token().type == TokenType.COMMA:
                self.advance()

        self.expect(TokenType.RPAREN)
        return ForeachClause(variable=variable, list_expr=list_expr, actions=actions)

    def _parse_create_index(self) -> CreateIndexClause:
        """Parse CREATE INDEX clause."""
        self.expect(TokenType.CREATE)
        unique = False
        if self.current_token().type == TokenType.UNIQUE:
            unique = True
            self.advance()
        self.expect(TokenType.INDEX)

        if_not_exists = False
        if self.current_token().type == TokenType.IF:
            self.advance()
            if self.current_token().type != TokenType.NOT:
                raise CypherSyntaxError(
                    f"Expected NOT after IF, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            self.advance()
            if self.current_token().type != TokenType.EXISTS:
                raise CypherSyntaxError(
                    f"Expected EXISTS after IF NOT, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            self.advance()
            if_not_exists = True

        if self.current_token().type == TokenType.ON:
            return self._parse_create_index_legacy(unique, if_not_exists)

        if self.current_token().type == TokenType.FOR and self.peek_token().type == TokenType.LPAREN:
            return self._parse_create_index_neo4j(unique, if_not_exists)

        name = None
        if self.current_token().type in self._name_tokens and self.peek_token().type == TokenType.FOR:
            name = self.current_token().value
            self.advance()

        self.expect(TokenType.FOR)
        if self.current_token().type == TokenType.NODE:
            entity = "node"
        elif self.current_token().type == TokenType.RELATIONSHIP:
            entity = "relationship"
        else:
            raise CypherSyntaxError(
                f"Expected NODE or RELATIONSHIP after FOR, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        self.advance()

        self.expect(TokenType.COLON)
        if self.current_token().type not in self._name_tokens:
            raise CypherSyntaxError(
                f"Expected label/type after ':', got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        label_or_type = self.current_token().value
        self.advance()

        self.expect(TokenType.LPAREN)
        if self.current_token().type not in self._name_tokens:
            raise CypherSyntaxError(
                f"Expected property name in index definition, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        property_name = self.current_token().value
        self.advance()
        self.expect(TokenType.RPAREN)

        return CreateIndexClause(
            entity=entity,
            label_or_type=label_or_type,
            property=property_name,
            name=name,
            unique=unique,
            if_not_exists=if_not_exists
        )

    def _parse_create_index_legacy(self, unique: bool, if_not_exists: bool) -> CreateIndexClause:
        """Parse legacy syntax: CREATE INDEX ON :Label(prop)."""
        self.expect(TokenType.ON)
        self.expect(TokenType.COLON)
        if self.current_token().type not in self._name_tokens:
            raise CypherSyntaxError(
                f"Expected label after ':', got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        label_or_type = self.current_token().value
        self.advance()
        self.expect(TokenType.LPAREN)
        if self.current_token().type not in self._name_tokens:
            raise CypherSyntaxError(
                f"Expected property name in index definition, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        property_name = self.current_token().value
        self.advance()
        self.expect(TokenType.RPAREN)
        return CreateIndexClause(
            entity="node",
            label_or_type=label_or_type,
            property=property_name,
            unique=unique,
            if_not_exists=if_not_exists
        )

    def _parse_create_index_neo4j(self, unique: bool, if_not_exists: bool) -> CreateIndexClause:
        """Parse Neo4j-style index syntax: FOR (n:Label) ON (n.prop)."""
        self.expect(TokenType.FOR)

        node = self._parse_node_pattern()
        if node.properties:
            raise CypherSyntaxError(
                "Index FOR pattern cannot include properties",
                self.current_token().line,
                self.current_token().column
            )

        entity = "node"
        label_or_type = None
        target_variable = node.variable

        if self.current_token().type in (TokenType.DASH, TokenType.ARROW_LEFT):
            rel = self._parse_relationship_pattern()
            self._parse_node_pattern()
            entity = "relationship"
            if rel.rel_type is None:
                raise CypherSyntaxError(
                    "Relationship index FOR pattern must include a type",
                    self.current_token().line,
                    self.current_token().column
                )
            label_or_type = rel.rel_type
            target_variable = rel.variable
        else:
            if not node.labels:
                raise CypherSyntaxError(
                    "Index FOR pattern must include a label",
                    self.current_token().line,
                    self.current_token().column
                )
            if len(node.labels) != 1:
                raise CypherSyntaxError(
                    "Index FOR pattern must include exactly one label",
                    self.current_token().line,
                    self.current_token().column
                )
            label_or_type = node.labels[0]

        self.expect(TokenType.ON)
        var_name, prop_name = self._parse_index_on_property()
        if target_variable is not None and var_name != target_variable:
            raise CypherSyntaxError(
                f"Index ON property must reference '{target_variable}'",
                self.current_token().line,
                self.current_token().column
            )

        return CreateIndexClause(
            entity=entity,
            label_or_type=label_or_type,
            property=prop_name,
            unique=unique,
            if_not_exists=if_not_exists
        )

    def _parse_index_on_property(self) -> tuple[str, str]:
        """Parse ON (var.prop) property reference."""
        self.expect(TokenType.LPAREN)
        if self.current_token().type != TokenType.IDENTIFIER:
            raise CypherSyntaxError(
                f"Expected variable name in index property, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        var_name = self.current_token().value
        self.advance()
        self.expect(TokenType.DOT)
        if self.current_token().type != TokenType.IDENTIFIER:
            raise CypherSyntaxError(
                f"Expected property name in index property, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        prop_name = self.current_token().value
        self.advance()
        self.expect(TokenType.RPAREN)
        return var_name, prop_name

    def _parse_drop_index(self) -> DropIndexClause:
        """Parse DROP INDEX clause."""
        self.expect(TokenType.DROP)
        self.expect(TokenType.INDEX)
        if_exists = False
        if self.current_token().type == TokenType.IF:
            self.advance()
            if self.current_token().type != TokenType.EXISTS:
                raise CypherSyntaxError(
                    f"Expected EXISTS after IF, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            self.advance()
            if_exists = True
        if self.current_token().type not in self._name_tokens:
            raise CypherSyntaxError(
                f"Expected index name after DROP INDEX, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        name = self.current_token().value
        self.advance()
        return DropIndexClause(name=name, if_exists=if_exists)

    def _parse_show_indexes(self) -> ShowIndexesClause:
        """Parse SHOW INDEXES clause."""
        self.expect(TokenType.SHOW)
        if self.current_token().type not in (TokenType.INDEXES, TokenType.INDEX):
            raise CypherSyntaxError(
                f"Expected INDEXES after SHOW, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        self.advance()
        where_expr = None
        if self.current_token().type == TokenType.WHERE:
            self.advance()
            where_expr = self._parse_expression()
        return ShowIndexesClause(where_expr=where_expr)

    def _parse_create_constraint(self) -> CreateConstraintClause:
        """Parse CREATE CONSTRAINT clause."""
        self.expect(TokenType.CREATE)
        self.expect(TokenType.CONSTRAINT)

        name = None
        if self.current_token().type in self._name_tokens and self.peek_token().type in (TokenType.IF, TokenType.FOR):
            name = self.current_token().value
            self.advance()

        if_not_exists = False
        if self.current_token().type == TokenType.IF:
            self.advance()
            if self.current_token().type != TokenType.NOT:
                raise CypherSyntaxError(
                    f"Expected NOT after IF, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            self.advance()
            if self.current_token().type != TokenType.EXISTS:
                raise CypherSyntaxError(
                    f"Expected EXISTS after IF NOT, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            self.advance()
            if_not_exists = True

        if self.current_token().type == TokenType.ON:
            return self._parse_create_constraint_legacy(name, if_not_exists)

        self.expect(TokenType.FOR)
        if self.current_token().type == TokenType.NODE:
            entity = "node"
            self.advance()
            self.expect(TokenType.COLON)
            if self.current_token().type not in self._name_tokens:
                raise CypherSyntaxError(
                    f"Expected label after ':', got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            label_or_type = self.current_token().value
            self.advance()
            target_variable = None
        elif self.current_token().type == TokenType.RELATIONSHIP:
            entity = "relationship"
            self.advance()
            self.expect(TokenType.COLON)
            if self.current_token().type not in self._name_tokens:
                raise CypherSyntaxError(
                    f"Expected type after ':', got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            label_or_type = self.current_token().value
            self.advance()
            target_variable = None
        elif self.current_token().type == TokenType.LPAREN:
            entity, label_or_type, target_variable = self._parse_constraint_pattern()
        else:
            raise CypherSyntaxError(
                f"Expected NODE, RELATIONSHIP, or '(', got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )

        if self.current_token().type == TokenType.REQUIRE:
            self.advance()
            prop_var, prop_name = self._parse_constraint_property()
        elif self.current_token().type == TokenType.ON:
            self.advance()
            prop_var, prop_name = self._parse_index_on_property()
        else:
            raise CypherSyntaxError(
                f"Expected REQUIRE or ON, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        if target_variable is not None and prop_var != target_variable:
            raise CypherSyntaxError(
                f"Constraint property must reference '{target_variable}'",
                self.current_token().line,
                self.current_token().column
            )

        self.expect(TokenType.IS)
        if self.current_token().type == TokenType.UNIQUE:
            constraint_type = "UNIQUE"
            type_name = None
            self.advance()
        elif self.current_token().type == TokenType.NOT:
            self.advance()
            if self.current_token().type != TokenType.NULL:
                raise CypherSyntaxError(
                    f"Expected NULL after IS NOT, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            self.advance()
            constraint_type = "EXISTS"
            type_name = None
        else:
            if self.current_token().type not in self._name_tokens:
                raise CypherSyntaxError(
                    f"Expected constraint type after IS, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            constraint_type = "TYPE"
            type_name = self.current_token().value.upper()
            self.advance()

        return CreateConstraintClause(
            entity=entity,
            label_or_type=label_or_type,
            property=prop_name,
            constraint_type=constraint_type,
            type_name=type_name,
            name=name,
            if_not_exists=if_not_exists
        )

    def _parse_create_constraint_legacy(
        self,
        name: str | None,
        if_not_exists: bool,
    ) -> CreateConstraintClause:
        """Parse legacy syntax: CREATE CONSTRAINT ON (n:Label) ASSERT n.prop IS UNIQUE."""
        self.expect(TokenType.ON)
        if self.current_token().type != TokenType.LPAREN:
            raise CypherSyntaxError(
                f"Expected '(' after ON, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        node = self._parse_node_pattern()
        if not node.labels:
            raise CypherSyntaxError(
                "Constraint ON pattern must include a label",
                self.current_token().line,
                self.current_token().column
            )
        if len(node.labels) != 1:
            raise CypherSyntaxError(
                "Constraint ON pattern must include exactly one label",
                self.current_token().line,
                self.current_token().column
            )
        label_or_type = node.labels[0]
        target_variable = node.variable
        self.expect(TokenType.ASSERT)
        prop_var, prop_name = self._parse_constraint_property()
        if target_variable is not None and prop_var != target_variable:
            raise CypherSyntaxError(
                f"Constraint property must reference '{target_variable}'",
                self.current_token().line,
                self.current_token().column
            )
        self.expect(TokenType.IS)
        if self.current_token().type != TokenType.UNIQUE:
            raise CypherSyntaxError(
                f"Expected UNIQUE after IS, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        self.advance()
        return CreateConstraintClause(
            entity="node",
            label_or_type=label_or_type,
            property=prop_name,
            constraint_type="UNIQUE",
            type_name=None,
            name=name,
            if_not_exists=if_not_exists
        )
    def _parse_constraint_pattern(self) -> tuple[str, str, Optional[str]]:
        """Parse Neo4j-style constraint pattern."""
        node = self._parse_node_pattern()
        if node.properties:
            raise CypherSyntaxError(
                "Constraint FOR pattern cannot include properties",
                self.current_token().line,
                self.current_token().column
            )

        entity = "node"
        label_or_type = None
        target_variable = node.variable

        if self.current_token().type in (TokenType.DASH, TokenType.ARROW_LEFT):
            rel = self._parse_relationship_pattern()
            self._parse_node_pattern()
            entity = "relationship"
            if rel.rel_type is None:
                raise CypherSyntaxError(
                    "Relationship constraint FOR pattern must include a type",
                    self.current_token().line,
                    self.current_token().column
                )
            label_or_type = rel.rel_type
            target_variable = rel.variable
        else:
            if not node.labels:
                raise CypherSyntaxError(
                    "Constraint FOR pattern must include a label",
                    self.current_token().line,
                    self.current_token().column
                )
            if len(node.labels) != 1:
                raise CypherSyntaxError(
                    "Constraint FOR pattern must include exactly one label",
                    self.current_token().line,
                    self.current_token().column
                )
            label_or_type = node.labels[0]

        return entity, label_or_type, target_variable

    def _parse_constraint_property(self) -> tuple[str | None, str]:
        """Parse constraint property reference."""
        if self.current_token().type == TokenType.LPAREN:
            self.advance()
            if self.current_token().type == TokenType.IDENTIFIER and self.peek_token().type == TokenType.DOT:
                var_name = self.current_token().value
                self.advance()
                self.expect(TokenType.DOT)
                if self.current_token().type != TokenType.IDENTIFIER:
                    raise CypherSyntaxError(
                        f"Expected property name after '.', got {self.current_token().type.name}",
                        self.current_token().line,
                        self.current_token().column
                    )
                prop_name = self.current_token().value
                self.advance()
                self.expect(TokenType.RPAREN)
                return var_name, prop_name
            if self.current_token().type not in self._name_tokens:
                raise CypherSyntaxError(
                    f"Expected property name after '(', got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            prop_name = self.current_token().value
            self.advance()
            self.expect(TokenType.RPAREN)
            return None, prop_name

        if self.current_token().type == TokenType.IDENTIFIER and self.peek_token().type == TokenType.DOT:
            var_name = self.current_token().value
            self.advance()
            self.expect(TokenType.DOT)
            if self.current_token().type != TokenType.IDENTIFIER:
                raise CypherSyntaxError(
                    f"Expected property name after '.', got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            prop_name = self.current_token().value
            self.advance()
            return var_name, prop_name
        if self.current_token().type not in self._name_tokens:
            raise CypherSyntaxError(
                f"Expected property name after REQUIRE, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        prop_name = self.current_token().value
        self.advance()
        return None, prop_name

    def _parse_drop_constraint(self) -> DropConstraintClause:
        """Parse DROP CONSTRAINT clause."""
        self.expect(TokenType.DROP)
        self.expect(TokenType.CONSTRAINT)
        if_exists = False
        if self.current_token().type == TokenType.IF:
            self.advance()
            if self.current_token().type != TokenType.EXISTS:
                raise CypherSyntaxError(
                    f"Expected EXISTS after IF, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            self.advance()
            if_exists = True
        if self.current_token().type not in self._name_tokens:
            raise CypherSyntaxError(
                f"Expected constraint name after DROP CONSTRAINT, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        name = self.current_token().value
        self.advance()
        return DropConstraintClause(name=name, if_exists=if_exists)

    def _parse_show_constraints(self) -> ShowConstraintsClause:
        """Parse SHOW CONSTRAINTS clause."""
        self.expect(TokenType.SHOW)
        self.expect(TokenType.CONSTRAINTS)
        where_expr = None
        if self.current_token().type == TokenType.WHERE:
            self.advance()
            where_expr = self._parse_expression()
        return ShowConstraintsClause(where_expr=where_expr)

    def _parse_unwind(self) -> UnwindClause:
        """Parse UNWIND clause: UNWIND expr AS variable."""
        self.expect(TokenType.UNWIND)
        list_expr = self._parse_expression()
        self.expect(TokenType.AS)
        if self.current_token().type != TokenType.IDENTIFIER:
            raise CypherSyntaxError(
                f"Expected identifier after AS, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        variable = self.current_token().value
        self.advance()
        return UnwindClause(list_expr=list_expr, variable=variable)

    def _parse_call(self) -> SubqueryClause | ProcedureCallClause:
        """Parse CALL { ... } subquery clause or CALL procedure."""
        self.expect(TokenType.CALL)

        if self.current_token().type == TokenType.LBRACE:
            self.expect(TokenType.LBRACE)

            start_pos = self.pos
            depth = 1

            while self.pos < len(self.tokens):
                token = self.current_token()
                if token.type == TokenType.LBRACE:
                    depth += 1
                elif token.type == TokenType.RBRACE:
                    depth -= 1
                    if depth == 0:
                        break
                self.advance()

            if depth != 0:
                raise CypherSyntaxError(
                    "Unterminated CALL subquery block",
                    self.current_token().line,
                    self.current_token().column
                )

            end_pos = self.pos
            sub_tokens = self.tokens[start_pos:end_pos]
            eof_token = Token(TokenType.EOF, None, self.current_token().line, self.current_token().column)
            sub_parser = Parser(sub_tokens + [eof_token])
            subquery = sub_parser.parse()

            self.expect(TokenType.RBRACE)

            return SubqueryClause(query=subquery)

        if self.current_token().type not in self._name_tokens:
            raise CypherSyntaxError(
                f"Expected procedure name after CALL, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        name_parts = [self.current_token().value]
        self.advance()
        while self.current_token().type == TokenType.DOT:
            self.advance()
            if self.current_token().type not in self._name_tokens:
                raise CypherSyntaxError(
                    f"Expected identifier after '.', got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            name_parts.append(self.current_token().value)
            self.advance()
        name = ".".join(name_parts)

        self.expect(TokenType.LPAREN)
        arguments = []
        if self.current_token().type != TokenType.RPAREN:
            while True:
                arguments.append(self._parse_expression())
                if self.current_token().type == TokenType.COMMA:
                    self.advance()
                    continue
                break
        self.expect(TokenType.RPAREN)

        yield_items = []
        if self.current_token().type == TokenType.YIELD:
            self.advance()
            while True:
                if self.current_token().type not in self._name_tokens:
                    raise CypherSyntaxError(
                        f"Expected identifier after YIELD, got {self.current_token().type.name}",
                        self.current_token().line,
                        self.current_token().column
                    )
                yield_items.append(self.current_token().value)
                self.advance()
                if self.current_token().type == TokenType.COMMA:
                    self.advance()
                    continue
                break

        return ProcedureCallClause(name=name, arguments=arguments, yield_items=yield_items)

    def _parse_create(self) -> CreateClause:
        """Parse CREATE clause: CREATE pattern [, pattern, ...]."""
        self.expect(TokenType.CREATE)
        patterns = [self._parse_pattern()]

        # Parse additional comma-separated patterns
        while self.current_token().type == TokenType.COMMA:
            self.advance()  # Skip comma
            patterns.append(self._parse_pattern())

        return CreateClause(patterns=patterns)

    def _parse_merge(self) -> MergeClause:
        """Parse MERGE clause: MERGE pattern [ON CREATE SET ...] [ON MATCH SET ...].

        Examples:
            MERGE (n:Person {email: 'alice@example.com'})
            MERGE (n:Person {email: 'alice@example.com'})
              ON CREATE SET n.created = 2024
              ON MATCH SET n.lastSeen = 2024
        """
        self.expect(TokenType.MERGE)
        patterns = [self._parse_pattern()]

        on_create_set = None
        on_match_set = None

        # Parse ON CREATE SET and ON MATCH SET
        while self.current_token().type == TokenType.ON:
            self.advance()  # consume ON

            if self.current_token().type == TokenType.CREATE:
                self.advance()  # consume CREATE
                self.expect(TokenType.SET)
                on_create_set = self._parse_set_items()
            elif self.current_token().type == TokenType.MATCH:
                self.advance()  # consume MATCH
                self.expect(TokenType.SET)
                on_match_set = self._parse_set_items()
            else:
                raise CypherSyntaxError(
                    f"Expected CREATE or MATCH after ON, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )

        return MergeClause(
            patterns=patterns,
            on_create_set=on_create_set,
            on_match_set=on_match_set
        )

    def _parse_match(self, optional: bool = False) -> MatchClause:
        """Parse MATCH clause: [OPTIONAL] MATCH pattern [WHERE] [DELETE|SET|REMOVE] [RETURN] [ORDER BY] [SKIP] [LIMIT]."""
        self.expect(TokenType.MATCH)
        patterns = [self._parse_pattern()]

        while self.current_token().type == TokenType.COMMA:
            self.advance()
            patterns.append(self._parse_pattern())

        where_clause = None
        if self.current_token().type == TokenType.WHERE:
            where_clause = self._parse_where()

        delete_clause = None
        set_clause = None
        remove_clause = None
        if self.current_token().type == TokenType.DELETE:
            delete_clause = self._parse_delete()
        elif self.current_token().type == TokenType.SET:
            set_clause = self._parse_set()
        elif self.current_token().type == TokenType.REMOVE:
            remove_clause = self._parse_remove()

        return_clause = None
        if self.current_token().type == TokenType.RETURN:
            return_clause = self._parse_return()

        order_by_clause = None
        if self.current_token().type == TokenType.ORDER:
            order_by_clause = self._parse_order_by()

        skip_clause = None
        if self.current_token().type == TokenType.SKIP:
            skip_clause = self._parse_skip()

        limit_clause = None
        if self.current_token().type == TokenType.LIMIT:
            limit_clause = self._parse_limit()

        return MatchClause(
            patterns=patterns,
            where_clause=where_clause,
            delete_clause=delete_clause,
            set_clause=set_clause,
            remove_clause=remove_clause,
            return_clause=return_clause,
            order_by_clause=order_by_clause,
            skip_clause=skip_clause,
            limit_clause=limit_clause,
            optional=optional
        )

    def _parse_pattern(self) -> Pattern:
        """Parse a pattern: (node) or (node)-[rel]->(node)."""
        elements = []
        variable = None

        if self.current_token().type == TokenType.IDENTIFIER and self.peek_token().type == TokenType.EQ:
            variable = self.current_token().value
            self.advance()
            self.expect(TokenType.EQ)

        if self.current_token().type in (TokenType.SHORTESTPATH, TokenType.ALLSHORTESTPATHS):
            func_token = self.current_token()
            self.advance()
            self.expect(TokenType.LPAREN)
            inner_pattern = self._parse_pattern()
            self.expect(TokenType.RPAREN)
            return PatternFunction(name=func_token.type.name, pattern=inner_pattern, variable=variable)

        # Parse first node
        node = self._parse_node_pattern()
        elements.append(PatternElement(node=node))

        # Parse relationship patterns if present
        while self.current_token().type in (TokenType.DASH, TokenType.ARROW_LEFT):
            relationship = self._parse_relationship_pattern()
            node = self._parse_node_pattern()
            # Update last element's relationship
            elements[-1].relationship = relationship
            # Add new node
            elements.append(PatternElement(node=node))

        return Pattern(variable=variable, elements=elements)

    def _parse_node_pattern(self) -> NodePattern:
        """Parse node pattern: (variable:Label1:Label2 {props})."""
        self.expect(TokenType.LPAREN)

        variable = None
        labels = []
        properties = {}

        # Check if there's a variable or label
        if self.current_token().type == TokenType.IDENTIFIER:
            variable = self.current_token().value
            self.advance()

        # Parse labels (: followed by identifier/keyword)
        while self.current_token().type == TokenType.COLON:
            self.advance()  # Skip :
            if not self._is_symbolic_name_token(self.current_token()):
                raise CypherSyntaxError(
                    f"Expected label name after ':', got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            labels.append(self.current_token().value)
            self.advance()

        # Parse properties if present
        if self.current_token().type == TokenType.LBRACE:
            properties = self._parse_properties()

        self.expect(TokenType.RPAREN)

        return NodePattern(
            variable=variable,
            labels=labels,
            properties=properties
        )

    def _parse_relationship_pattern(self) -> RelationshipPattern:
        """Parse relationship pattern: -[r:TYPE {props}]-> or <-[r:TYPE]-."""
        # Determine start direction
        starts_with_left_arrow = False
        if self.current_token().type == TokenType.ARROW_LEFT:
            starts_with_left_arrow = True
            self.advance()  # Skip <-
        elif self.current_token().type == TokenType.DASH:
            self.advance()  # Skip -
        else:
            raise CypherSyntaxError(
                f"Expected '-' or '<-' in relationship pattern",
                self.current_token().line,
                self.current_token().column
            )

        variable = None
        rel_type = None
        properties = {}
        min_hops = 1
        max_hops = 1

        # Parse [r:TYPE {props}] if present
        if self.current_token().type == TokenType.LBRACKET:
            self.advance()  # Skip [

            # Variable
            if self.current_token().type == TokenType.IDENTIFIER:
                variable = self.current_token().value
                self.advance()

            # Type
            if self.current_token().type == TokenType.COLON:
                self.advance()  # Skip :
                if not self._is_symbolic_name_token(self.current_token()):
                    raise CypherSyntaxError(
                        f"Expected relationship type after ':', got {self.current_token().type.name}",
                        self.current_token().line,
                        self.current_token().column
                    )
                rel_type = self.current_token().value
                self.advance()

            # Length (variable-length paths)
            if self.current_token().type == TokenType.ASTERISK:
                min_hops, max_hops = self._parse_relationship_length()

            # Properties
            if self.current_token().type == TokenType.LBRACE:
                properties = self._parse_properties()

            self.expect(TokenType.RBRACKET)

        # Determine end direction
        ends_with_right_arrow = False
        if self.current_token().type == TokenType.ARROW_RIGHT:
            ends_with_right_arrow = True
            self.advance()  # Skip ->
        elif self.current_token().type == TokenType.DASH:
            self.advance()  # Skip -
        else:
            raise CypherSyntaxError(
                f"Expected '-' or '->' at end of relationship pattern",
                self.current_token().line,
                self.current_token().column
            )

        # Determine final direction based on arrows
        if starts_with_left_arrow and not ends_with_right_arrow:
            direction = 'incoming'  # <-[r]-
        elif not starts_with_left_arrow and ends_with_right_arrow:
            direction = 'outgoing'  # -[r]->
        else:
            direction = 'both'  # -[r]- or <-[r]->

        return RelationshipPattern(
            variable=variable,
            rel_type=rel_type,
            properties=properties,
            direction=direction,
            min_hops=min_hops,
            max_hops=max_hops
        )

    def _parse_relationship_length(self) -> tuple[int, int | None]:
        """Parse relationship length: *n, *min..max, *min.., *..max, or *."""
        self.expect(TokenType.ASTERISK)

        min_hops = None
        max_hops = None

        if self.current_token().type == TokenType.INTEGER:
            min_hops = self.current_token().value
            self.advance()

        # Check for range ".."
        saw_range = False
        if self.current_token().type == TokenType.DOT and self.peek_token().type == TokenType.DOT:
            saw_range = True
            self.advance()
            self.advance()
            if self.current_token().type == TokenType.INTEGER:
                max_hops = self.current_token().value
                self.advance()

        if min_hops is None and max_hops is None:
            return 1, None
        if min_hops is not None and max_hops is None and not saw_range:
            return min_hops, min_hops
        if min_hops is None and max_hops is not None:
            return 1, max_hops

        return min_hops, max_hops

    def _parse_properties(self) -> dict:
        """Parse property map: {key: value, key2: value2}."""
        self.expect(TokenType.LBRACE)

        properties = {}

        while self.current_token().type != TokenType.RBRACE:
            # Parse key (identifier)
            if self.current_token().type != TokenType.IDENTIFIER:
                raise CypherSyntaxError(
                    f"Expected property name, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            key = self.current_token().value
            self.advance()

            # Expect colon
            self.expect(TokenType.COLON)

            # Parse value (expression)
            value = self._parse_expression()
            properties[key] = value

            # Check for comma
            if self.current_token().type == TokenType.COMMA:
                self.advance()
            elif self.current_token().type != TokenType.RBRACE:
                raise CypherSyntaxError(
                    f"Expected ',' or '}}' in properties, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )

        self.expect(TokenType.RBRACE)
        return properties

    def _is_symbolic_name_token(self, token: Token) -> bool:
        """Return True if the token can be used as a label/type name."""
        if token.type == TokenType.IDENTIFIER:
            return True
        if token.type in (TokenType.BOOLEAN, TokenType.NULL):
            return False
        return token.type in self._symbolic_name_tokens

    def _parse_literal_value(self) -> any:
        """Parse a literal value (string, number, boolean, null, list, map)."""
        expr = self._parse_expression()
        return self._expression_to_literal(expr)

    def _expression_to_literal(self, expr: Expression) -> any:
        """Convert literal/list/map expressions into concrete values."""
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, UnaryOp):
            if expr.operator == '+' and isinstance(expr.operand, Literal):
                return expr.operand.value
            if expr.operator == '-' and isinstance(expr.operand, Literal):
                return -expr.operand.value
        if isinstance(expr, ListLiteral):
            return [self._expression_to_literal(item) for item in expr.items]
        if isinstance(expr, MapLiteral):
            return {key: self._expression_to_literal(value) for key, value in expr.items.items()}
        token = self.current_token()
        raise CypherSyntaxError(
            f"Expected literal value, got {type(expr).__name__}",
            token.line,
            token.column
        )

    def _parse_where(self) -> WhereClause:
        """Parse WHERE clause: WHERE expression."""
        self.expect(TokenType.WHERE)
        condition = self._parse_expression()
        return WhereClause(condition=condition)

    def _parse_expression(self) -> Expression:
        """Parse expression with operator precedence.

        Precedence (lowest to highest):
        1. OR
        2. AND
        3. NOT
        4. Comparison (=, !=, <, >, <=, >=)
        """
        return self._parse_or_expression()

    def _parse_or_expression(self) -> Expression:
        """Parse OR expression."""
        left = self._parse_and_expression()

        while self.current_token().type == TokenType.OR:
            self.advance()
            right = self._parse_and_expression()
            left = BinaryOp(left=left, operator='OR', right=right)

        return left

    def _parse_and_expression(self) -> Expression:
        """Parse AND expression."""
        left = self._parse_not_expression()

        while self.current_token().type == TokenType.AND:
            self.advance()
            right = self._parse_not_expression()
            left = BinaryOp(left=left, operator='AND', right=right)

        return left

    def _parse_not_expression(self) -> Expression:
        """Parse NOT expression."""
        if self.current_token().type == TokenType.NOT:
            self.advance()
            operand = self._parse_not_expression()
            return UnaryOp(operator='NOT', operand=operand)

        return self._parse_comparison_expression()

    def _parse_comparison_expression(self) -> Expression:
        """Parse comparison expression."""
        left = self._parse_additive_expression()

        # Label predicate: `n:Label` (or `n:A:B`) as a boolean test.
        if self.current_token().type == TokenType.COLON:
            labels = []
            while self.current_token().type == TokenType.COLON:
                self.advance()
                if self.current_token().type not in self._name_tokens:
                    raise CypherSyntaxError(
                        f"Expected label name after ':', got {self.current_token().type.name}",
                        self.current_token().line,
                        self.current_token().column
                    )
                labels.append(self.current_token().value)
                self.advance()
            return LabelPredicate(operand=left, labels=labels)

        # Check for comparison operators
        token = self.current_token()
        if token.type == TokenType.IS:
            self.advance()
            is_not = False
            if self.current_token().type == TokenType.NOT:
                is_not = True
                self.advance()
            if self.current_token().type != TokenType.NULL:
                raise CypherSyntaxError(
                    f"Expected NULL after IS, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            self.advance()
            operator = 'IS NOT NULL' if is_not else 'IS NULL'
            return UnaryOp(operator=operator, operand=left)
        if token.type in (TokenType.EQ, TokenType.NEQ, TokenType.LT,
                          TokenType.GT, TokenType.LTE, TokenType.GTE, TokenType.IN):
            operator = token.value
            if token.type == TokenType.IN:
                operator = 'IN'
            if token.type == TokenType.NEQ:
                operator = '!='
            self.advance()
            right = self._parse_additive_expression()
            return BinaryOp(left=left, operator=operator, right=right)

        return left

    def _parse_additive_expression(self) -> Expression:
        """Parse additive expression ('+' and '-')."""
        left = self._parse_unary_expression()

        while self.current_token().type in (TokenType.PLUS, TokenType.DASH):
            operator_token = self.current_token()
            self.advance()
            right = self._parse_unary_expression()
            left = BinaryOp(left=left, operator=operator_token.value, right=right)

        return left

    def _parse_unary_expression(self) -> Expression:
        """Parse unary expressions ('+' and '-')."""
        token = self.current_token()
        if token.type == TokenType.PLUS:
            self.advance()
            return self._parse_unary_expression()
        if token.type == TokenType.DASH:
            self.advance()
            operand = self._parse_unary_expression()
            return UnaryOp(operator='-', operand=operand)
        return self._parse_primary_expression()

    def _parse_primary_expression(self) -> Expression:
        """Parse primary expression (literal, property access, function call, parenthesized)."""
        expr = self._parse_primary_base()

        # Handle postfix operators: list indexing and property access.
        while self.current_token().type in (TokenType.LBRACKET, TokenType.DOT):
            if self.current_token().type == TokenType.LBRACKET:
                expr = self._parse_list_index_or_slice(expr)
                continue
            if self.current_token().type == TokenType.DOT and self.peek_token().type == TokenType.DOT:
                break

            self.advance()
            if self.current_token().type not in self._name_tokens:
                raise CypherSyntaxError(
                    f"Expected property name after '.', got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            property_name = self.current_token().value
            self.advance()
            expr = PropertyLookup(base_expr=expr, property=property_name)

        return expr

    def _parse_primary_base(self) -> Expression:
        """Parse base primary expression without postfix operators."""
        token = self.current_token()

        # CASE expressions
        if token.type == TokenType.CASE:
            return self._parse_case_expression()

        # Aggregation functions: COUNT, SUM, AVG, MIN, MAX, COLLECT, STDDEV, PERCENTILECONT.
        # Only treat as a function call when followed by '('; otherwise the keyword is being
        # used as a plain name (e.g. an alias referenced in ORDER BY: `... AS count ORDER BY count`),
        # so fall through to the generic name handling below.
        if token.type in (TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX,
                          TokenType.COLLECT, TokenType.STDDEV, TokenType.PERCENTILECONT):
            if self.peek_token().type == TokenType.LPAREN:
                return self._parse_function_call()

        # Parenthesized expression
        if token.type == TokenType.LPAREN:
            self.advance()
            expr = self._parse_expression()
            self.expect(TokenType.RPAREN)
            return expr

        # List predicates: ANY/ALL/NONE/SINGLE
        if token.type in (TokenType.ANY, TokenType.ALL, TokenType.NONE, TokenType.SINGLE):
            return self._parse_list_predicate()

        # List literals / comprehensions
        if token.type == TokenType.LBRACKET:
            return self._parse_list_expression()

        # Map literals
        if token.type == TokenType.LBRACE:
            return self._parse_map_literal()

        # Property access (variable.property) or generic function call
        if token.type in self._name_tokens:
            variable = token.value
            self.advance()

            segments = [variable]
            while self.current_token().type == TokenType.DOT:
                self.advance()
                if self.current_token().type not in self._name_tokens:
                    raise CypherSyntaxError(
                        f"Expected property name after '.', got {self.current_token().type.name}",
                        self.current_token().line,
                        self.current_token().column
                    )
                segments.append(self.current_token().value)
                self.advance()

            if self.current_token().type == TokenType.LPAREN:
                if len(segments) == 1:
                    func_name = segments[0].lower()
                    if func_name == 'filter':
                        return self._parse_filter_expression()
                    if func_name == 'extract':
                        return self._parse_extract_expression()
                    if func_name == 'reduce':
                        return self._parse_reduce_expression()
                return self._parse_generic_function_call(".".join(segments))

            if len(segments) == 1:
                return Variable(name=segments[0])
            if len(segments) == 2:
                return PropertyAccess(variable=segments[0], property=segments[1])
            expr: Expression = PropertyAccess(variable=segments[0], property=segments[1])
            for prop_name in segments[2:]:
                expr = PropertyLookup(base_expr=expr, property=prop_name)
            return expr

        # Literal values
        if token.type in (TokenType.STRING, TokenType.INTEGER, TokenType.FLOAT,
                          TokenType.BOOLEAN, TokenType.NULL):
            value = token.value
            self.advance()
            return Literal(value=value)

        raise CypherSyntaxError(
            f"Unexpected token in expression: {token.type.name}",
            token.line,
            token.column
        )

    def _parse_list_generator(self) -> tuple[str, Expression, Optional[Expression]]:
        """Parse list generator: var IN list [WHERE expr]."""
        if self.current_token().type != TokenType.IDENTIFIER:
            raise CypherSyntaxError(
                f"Expected identifier in list generator, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        var_name = self.current_token().value
        self.advance()
        self.expect(TokenType.IN)

        list_expr = self._parse_expression()
        where_expr = None
        if self.current_token().type == TokenType.WHERE:
            self.advance()
            where_expr = self._parse_expression()

        return var_name, list_expr, where_expr

    def _parse_filter_expression(self) -> Expression:
        """Parse filter expression: filter(var IN list WHERE expr)."""
        self.expect(TokenType.LPAREN)
        var_name, list_expr, where_expr = self._parse_list_generator()
        self.expect(TokenType.RPAREN)
        return ListComprehension(
            variable=var_name,
            list_expr=list_expr,
            projection=Variable(name=var_name),
            where_expr=where_expr
        )

    def _parse_extract_expression(self) -> Expression:
        """Parse extract expression: extract(var IN list [WHERE expr] | expr)."""
        self.expect(TokenType.LPAREN)
        var_name, list_expr, where_expr = self._parse_list_generator()
        self.expect(TokenType.PIPE)
        projection = self._parse_expression()
        self.expect(TokenType.RPAREN)
        return ListComprehension(
            variable=var_name,
            list_expr=list_expr,
            projection=projection,
            where_expr=where_expr
        )

    def _parse_reduce_expression(self) -> Expression:
        """Parse reduce expression: reduce(acc = init, var IN list | expr)."""
        self.expect(TokenType.LPAREN)

        if self.current_token().type != TokenType.IDENTIFIER:
            raise CypherSyntaxError(
                f"Expected accumulator name, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        acc_name = self.current_token().value
        self.advance()
        self.expect(TokenType.EQ)
        init_expr = self._parse_expression()
        self.expect(TokenType.COMMA)

        var_name, list_expr, _ = self._parse_list_generator()
        self.expect(TokenType.PIPE)
        expr = self._parse_expression()
        self.expect(TokenType.RPAREN)

        return ReduceExpression(
            accumulator=acc_name,
            init_expr=init_expr,
            variable=var_name,
            list_expr=list_expr,
            expression=expr
        )

    def _parse_map_literal(self) -> MapLiteral:
        """Parse map literal: {key: value, ...}."""
        self.expect(TokenType.LBRACE)
        items: dict[str, Expression] = {}

        while self.current_token().type != TokenType.RBRACE:
            key_token = self.current_token()
            if key_token.type == TokenType.IDENTIFIER:
                key = key_token.value
                self.advance()
            elif key_token.type in (TokenType.OR, TokenType.AND):
                key = key_token.value.lower()
                self.advance()
            elif key_token.type == TokenType.STRING:
                key = key_token.value
                self.advance()
            else:
                raise CypherSyntaxError(
                    f"Expected map key, got {key_token.type.name}",
                    key_token.line,
                    key_token.column
                )

            self.expect(TokenType.COLON)
            value = self._parse_expression()
            items[key] = value

            if self.current_token().type == TokenType.COMMA:
                self.advance()
            elif self.current_token().type != TokenType.RBRACE:
                raise CypherSyntaxError(
                    f"Expected ',' or '}}' in map literal, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )

        self.expect(TokenType.RBRACE)
        return MapLiteral(items=items)

    def _parse_generic_function_call(self, name: str) -> FunctionCallExpression:
        """Parse a generic function call with arguments."""
        self.expect(TokenType.LPAREN)

        args = []
        if self.current_token().type != TokenType.RPAREN:
            while True:
                args.append(self._parse_expression())
                if self.current_token().type == TokenType.COMMA:
                    self.advance()
                    continue
                break

        self.expect(TokenType.RPAREN)
        return FunctionCallExpression(name=name, arguments=args)

    def _parse_list_expression(self) -> Expression:
        """Parse list literal or list comprehension."""
        self.expect(TokenType.LBRACKET)

        if self.current_token().type == TokenType.RBRACKET:
            self.advance()
            return ListLiteral(items=[])

        # List comprehension: [var IN list [WHERE cond] | expr]
        if self.current_token().type == TokenType.IDENTIFIER and self.peek_token().type == TokenType.IN:
            var_name, list_expr, where_expr = self._parse_list_generator()
            self.expect(TokenType.PIPE)
            projection = self._parse_expression()
            self.expect(TokenType.RBRACKET)

            return ListComprehension(
                variable=var_name,
                list_expr=list_expr,
                projection=projection,
                where_expr=where_expr
            )

        # Pattern comprehension: [pattern [WHERE cond] | expr]
        if self.current_token().type in (TokenType.LPAREN, TokenType.IDENTIFIER):
            start_pos = self.pos
            if self.current_token().type == TokenType.IDENTIFIER and self.peek_token().type != TokenType.EQ:
                start_pos = None
            if start_pos is not None:
                try:
                    pattern = self._parse_pattern()
                    where_expr = None
                    if self.current_token().type == TokenType.WHERE:
                        self.advance()
                        where_expr = self._parse_expression()
                    if self.current_token().type == TokenType.PIPE:
                        self.advance()
                        projection = self._parse_expression()
                        self.expect(TokenType.RBRACKET)
                        return PatternComprehension(
                            pattern=pattern,
                            projection=projection,
                            where_expr=where_expr
                        )
                except CypherSyntaxError:
                    pass
                self.pos = start_pos

        # List literal
        items = []
        while True:
            items.append(self._parse_expression())
            if self.current_token().type == TokenType.COMMA:
                self.advance()
                continue
            break

        self.expect(TokenType.RBRACKET)
        return ListLiteral(items=items)

    def _parse_list_index_or_slice(self, list_expr: Expression) -> Expression:
        """Parse list indexing or slicing."""
        self.expect(TokenType.LBRACKET)

        # Slice with omitted start: [..end]
        if self.current_token().type == TokenType.DOT and self.peek_token().type == TokenType.DOT:
            self.advance()
            self.advance()
            end_expr = None
            if self.current_token().type != TokenType.RBRACKET:
                end_expr = self._parse_expression()
            self.expect(TokenType.RBRACKET)
            return ListSlice(list_expr=list_expr, start_expr=None, end_expr=end_expr)

        start_expr = self._parse_expression()

        # Slice if ".." follows start
        if self.current_token().type == TokenType.DOT and self.peek_token().type == TokenType.DOT:
            self.advance()
            self.advance()
            end_expr = None
            if self.current_token().type != TokenType.RBRACKET:
                end_expr = self._parse_expression()
            self.expect(TokenType.RBRACKET)
            return ListSlice(list_expr=list_expr, start_expr=start_expr, end_expr=end_expr)

        self.expect(TokenType.RBRACKET)
        return ListIndex(list_expr=list_expr, index_expr=start_expr)

    def _parse_list_predicate(self) -> ListPredicate:
        """Parse list predicates: ANY/ALL/NONE/SINGLE (var IN list WHERE cond)."""
        token = self.current_token()
        predicate = token.type.name
        self.advance()
        self.expect(TokenType.LPAREN)

        if self.current_token().type != TokenType.IDENTIFIER:
            raise CypherSyntaxError(
                f"Expected identifier in list predicate, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        var_name = self.current_token().value
        self.advance()
        self.expect(TokenType.IN)
        list_expr = self._parse_expression()

        where_expr = None
        if self.current_token().type == TokenType.WHERE:
            self.advance()
            where_expr = self._parse_expression()

        self.expect(TokenType.RPAREN)

        return ListPredicate(
            predicate=predicate,
            variable=var_name,
            list_expr=list_expr,
            where_expr=where_expr
        )

    def _parse_case_expression(self) -> CaseExpression:
        """Parse CASE expression (searched or simple)."""
        self.expect(TokenType.CASE)

        base_expr = None
        whens = []
        else_expr = None

        if self.current_token().type != TokenType.WHEN:
            base_expr = self._parse_primary_expression()

        if self.current_token().type != TokenType.WHEN:
            raise CypherSyntaxError(
                f"Expected WHEN in CASE expression, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )

        while self.current_token().type == TokenType.WHEN:
            self.advance()
            when_expr = self._parse_expression()
            self.expect(TokenType.THEN)
            then_expr = self._parse_expression()
            whens.append(CaseWhen(when_expr=when_expr, then_expr=then_expr))

        if self.current_token().type == TokenType.ELSE:
            self.advance()
            else_expr = self._parse_expression()

        self.expect(TokenType.END)

        return CaseExpression(base_expr=base_expr, whens=whens, else_expr=else_expr)

    def _parse_function_call(self) -> FunctionCall:
        """Parse aggregation function call: COUNT(n), SUM(n.age), etc."""
        function_token = self.current_token()
        function_name = function_token.type.name  # 'COUNT', 'SUM', etc.
        self.advance()

        # Expect opening parenthesis
        self.expect(TokenType.LPAREN)

        distinct = False
        if self.current_token().type == TokenType.DISTINCT:
            distinct = True
            self.advance()

        # Parse arguments
        arguments = []
        star = False
        if self.current_token().type == TokenType.ASTERISK:
            star = True
            self.advance()
        elif self.current_token().type != TokenType.RPAREN:
            arguments.append(self._parse_expression())
            while self.current_token().type == TokenType.COMMA:
                self.advance()
                arguments.append(self._parse_expression())

        # Expect closing parenthesis
        self.expect(TokenType.RPAREN)

        return FunctionCall(
            function_name=function_name,
            arguments=arguments,
            star=star,
            distinct=distinct
        )

    def _parse_return(self) -> ReturnClause:
        """Parse RETURN clause: RETURN expr, expr, ...

        Supports:
        - Variables: RETURN n
        - Property access: RETURN n.name
        - Aggregations: RETURN COUNT(*), SUM(n.age), etc.
        """
        self.expect(TokenType.RETURN)
        distinct = False
        if self.current_token().type == TokenType.DISTINCT:
            distinct = True
            self.advance()

        items = []

        while True:
            # Parse return expression with full operator support
            expr = self._parse_expression()

            alias = None
            if self.current_token().type == TokenType.AS:
                self.advance()
                if self.current_token().type not in self._name_tokens:
                    raise CypherSyntaxError(
                        f"Expected identifier after AS, got {self.current_token().type.name}",
                        self.current_token().line,
                        self.current_token().column
                    )
                alias = self.current_token().value
                self.advance()

            items.append(ReturnItem(expression=expr, alias=alias))

            # Check for comma
            if self.current_token().type == TokenType.COMMA:
                self.advance()
            else:
                break

        return ReturnClause(items=items, distinct=distinct)

    def _parse_delete(self) -> DeleteClause:
        """Parse DELETE clause: DELETE n, r, m."""
        self.expect(TokenType.DELETE)

        variables = []
        while True:
            if self.current_token().type != TokenType.IDENTIFIER:
                raise CypherSyntaxError(
                    f"Expected variable in DELETE clause, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )

            variables.append(self.current_token().value)
            self.advance()

            # Check for comma
            if self.current_token().type == TokenType.COMMA:
                self.advance()
            else:
                break

        return DeleteClause(variables=variables)

    def _parse_set(self) -> SetClause:
        """Parse SET clause: SET n.prop1 = value1, n.prop2 = value2."""
        self.expect(TokenType.SET)
        return self._parse_set_items()

    def _parse_set_items(self) -> SetClause:
        """Parse SET items without expecting SET token."""
        items = []
        while True:
            # Parse variable.property
            if self.current_token().type != TokenType.IDENTIFIER:
                raise CypherSyntaxError(
                    f"Expected variable in SET clause, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )

            variable = self.current_token().value
            self.advance()

            if self.current_token().type == TokenType.DOT:
                self.advance()

                if self.current_token().type != TokenType.IDENTIFIER:
                    raise CypherSyntaxError(
                        f"Expected property name after '.', got {self.current_token().type.name}",
                        self.current_token().line,
                        self.current_token().column
                    )

                property_name = self.current_token().value
                self.advance()

                # Expect =
                self.expect(TokenType.EQ)

                # Parse value expression
                value = self._parse_expression()

                items.append(SetItem(variable=variable, property=property_name, value=value, operator="="))
            elif self.current_token().type == TokenType.PLUS and self.peek_token().type == TokenType.EQ:
                self.advance()
                self.advance()
                value = self._parse_expression()
                items.append(SetItem(variable=variable, property=None, value=value, operator="+="))
            elif self.current_token().type == TokenType.EQ:
                self.advance()
                value = self._parse_expression()
                items.append(SetItem(variable=variable, property=None, value=value, operator="="))
            else:
                raise CypherSyntaxError(
                    f"Expected '.', '+=' or '=' in SET clause, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )

            # Check for comma
            if self.current_token().type == TokenType.COMMA:
                self.advance()
            else:
                break

        return SetClause(items=items)

    def _parse_load_csv(self) -> LoadCsvClause:
        """Parse LOAD CSV clause."""
        self.expect(TokenType.LOAD)
        self.expect(TokenType.CSV)

        with_headers = False
        if self.current_token().type == TokenType.WITH:
            self.advance()
            self.expect(TokenType.HEADERS)
            with_headers = True

        self.expect(TokenType.FROM)
        source = self._parse_expression()

        self.expect(TokenType.AS)
        if self.current_token().type != TokenType.IDENTIFIER:
            raise CypherSyntaxError(
                f"Expected identifier after AS, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )
        variable = self.current_token().value
        self.advance()

        return LoadCsvClause(source=source, variable=variable, with_headers=with_headers)

    def _parse_remove(self) -> RemoveClause:
        """Parse REMOVE clause: REMOVE n.property, n:Label, ...

        Examples:
            REMOVE n.age              -- Remove property
            REMOVE n:OldLabel         -- Remove label
            REMOVE n.prop1, n.prop2   -- Remove multiple properties
        """
        self.expect(TokenType.REMOVE)

        items = []
        while True:
            # Parse variable
            if self.current_token().type != TokenType.IDENTIFIER:
                raise CypherSyntaxError(
                    f"Expected variable in REMOVE clause, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )
            variable = self.current_token().value
            self.advance()

            # Check for property (.property) or label (:Label)
            if self.current_token().type == TokenType.DOT:
                # Remove property: n.property
                self.advance()  # Skip .
                if self.current_token().type != TokenType.IDENTIFIER:
                    raise CypherSyntaxError(
                        f"Expected property name after '.', got {self.current_token().type.name}",
                        self.current_token().line,
                        self.current_token().column
                    )
                property = self.current_token().value
                self.advance()
                items.append(RemoveItem(variable=variable, property=property))
            elif self.current_token().type == TokenType.COLON:
                # Remove label: n:Label
                self.advance()  # Skip :
                if self.current_token().type != TokenType.IDENTIFIER:
                    raise CypherSyntaxError(
                        f"Expected label name after ':', got {self.current_token().type.name}",
                        self.current_token().line,
                        self.current_token().column
                    )
                label = self.current_token().value
                self.advance()
                items.append(RemoveItem(variable=variable, label=label))
            else:
                raise CypherSyntaxError(
                    f"Expected '.' or ':' after variable in REMOVE clause, got {self.current_token().type.name}",
                    self.current_token().line,
                    self.current_token().column
                )

            # Check for comma
            if self.current_token().type == TokenType.COMMA:
                self.advance()
            else:
                break

        return RemoveClause(items=items)

    def _parse_order_by(self) -> OrderByClause:
        """Parse ORDER BY clause: ORDER BY n.age ASC, n.name DESC."""
        self.expect(TokenType.ORDER)
        self.expect(TokenType.BY)

        items = []
        while True:
            # Parse expression (property access or variable)
            expr = self._parse_primary_expression()

            # Check for ASC/DESC
            ascending = True
            if self.current_token().type == TokenType.ASC:
                ascending = True
                self.advance()
            elif self.current_token().type == TokenType.DESC:
                ascending = False
                self.advance()

            items.append(OrderByItem(expression=expr, ascending=ascending))

            # Check for comma
            if self.current_token().type == TokenType.COMMA:
                self.advance()
            else:
                break

        return OrderByClause(items=items)

    def _parse_limit(self) -> LimitClause:
        """Parse LIMIT clause: LIMIT 10."""
        self.expect(TokenType.LIMIT)

        if self.current_token().type != TokenType.INTEGER:
            raise CypherSyntaxError(
                f"Expected integer in LIMIT clause, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )

        count = self.current_token().value
        self.advance()

        return LimitClause(count=count)

    def _parse_skip(self) -> SkipClause:
        """Parse SKIP clause: SKIP n."""
        self.expect(TokenType.SKIP)

        if self.current_token().type != TokenType.INTEGER:
            raise CypherSyntaxError(
                f"Expected integer in SKIP clause, got {self.current_token().type.name}",
                self.current_token().line,
                self.current_token().column
            )

        count = self.current_token().value
        self.advance()

        return SkipClause(count=count)

    def _parse_with(self) -> WithClause:
        """Parse WITH clause: WITH items [WHERE] [ORDER BY] [SKIP] [LIMIT].

        Examples:
            WITH n, n.age AS age
            WITH n, COUNT(n.friends) AS friendCount WHERE friendCount > 5
            WITH n ORDER BY n.age LIMIT 10
        """
        self.expect(TokenType.WITH)
        distinct = False
        if self.current_token().type == TokenType.DISTINCT:
            distinct = True
            self.advance()

        # Parse return items (similar to RETURN)
        items = []
        while True:
            # Parse expression
            expr = self._parse_primary_expression()

            # Check for AS alias
            alias = None
            if self.current_token().type == TokenType.AS:
                self.advance()
                if self.current_token().type not in self._name_tokens:
                    raise CypherSyntaxError(
                        f"Expected identifier after AS, got {self.current_token().type.name}",
                        self.current_token().line,
                        self.current_token().column
                    )
                alias = self.current_token().value
                self.advance()

            items.append(ReturnItem(expression=expr, alias=alias))

            # Check for comma
            if self.current_token().type == TokenType.COMMA:
                self.advance()
            else:
                break

        # Parse optional WHERE, ORDER BY, SKIP, LIMIT, RETURN
        where_clause = None
        return_clause = None
        order_by_clause = None
        skip_clause = None
        limit_clause = None

        while self.current_token().type in (TokenType.WHERE, TokenType.ORDER,
                                            TokenType.SKIP, TokenType.LIMIT, TokenType.RETURN):
            if self.current_token().type == TokenType.WHERE:
                where_clause = self._parse_where()
            elif self.current_token().type == TokenType.ORDER:
                order_by_clause = self._parse_order_by()
            elif self.current_token().type == TokenType.SKIP:
                skip_clause = self._parse_skip()
            elif self.current_token().type == TokenType.LIMIT:
                limit_clause = self._parse_limit()
            elif self.current_token().type == TokenType.RETURN:
                return_clause = self._parse_return()

        return WithClause(
            items=items,
            where_clause=where_clause,
            return_clause=return_clause,
            order_by_clause=order_by_clause,
            skip_clause=skip_clause,
            limit_clause=limit_clause,
            distinct=distinct
        )
