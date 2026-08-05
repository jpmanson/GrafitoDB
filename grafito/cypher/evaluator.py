"""Evaluator for WHERE clause expressions."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import re
import unicodedata
from typing import Any
from .ast_nodes import (
    Expression, Literal, Parameter, PropertyAccess, PropertyLookup, BinaryOp, UnaryOp, LabelPredicate,
    CaseExpression, ListLiteral, ListComprehension, ListIndex,
    ListSlice, ListPredicate, FunctionCallExpression, Variable, MapLiteral,
    ReduceExpression, PatternComprehension
)
from .exceptions import CypherExecutionError
from ..models import Path, Point

#: Threshold SIMILAR() applies when the caller gives none. Calibrated for
#: cosine similarity; inner-product indexes on unnormalised vectors should pass
#: min_score explicitly, and l2 indexes are required to (see _default_min_score).
DEFAULT_SIMILAR_MIN_SCORE = 0.5


class ExpressionEvaluator:
    """Evaluates WHERE clause expressions against a variable context."""

    def __init__(self, context: dict[str, Any], pattern_matcher=None, parameters: dict[str, Any] | None = None,
                 node_resolver=None, vector_scorer=None, vector_metric=None):
        """Initialize evaluator with variable context.

        Args:
            context: Dictionary mapping variable names to Node/Relationship objects
            pattern_matcher: Optional matcher for pattern comprehensions
            parameters: Optional query parameters, resolving $name references
            node_resolver: Optional callable(node_id) -> Node, for startNode/endNode
            vector_scorer: Optional callable(node, query, index) -> float | None,
                for SIMILAR()/VECTOR_SCORE()
            vector_metric: Optional callable(index) -> str, naming the index metric
        """
        self.context = context
        self.pattern_matcher = pattern_matcher
        self.parameters = parameters or {}
        self.node_resolver = node_resolver
        self.vector_scorer = vector_scorer
        self.vector_metric = vector_metric

    def evaluate(self, expr: Expression) -> Any:
        """Evaluate an expression and return the result.

        Args:
            expr: Expression AST node to evaluate

        Returns:
            The evaluated result (bool, int, str, etc.)

        Raises:
            CypherExecutionError: If evaluation fails
        """
        if isinstance(expr, Literal):
            return self._evaluate_literal(expr)
        elif isinstance(expr, Parameter):
            return self._evaluate_parameter(expr)
        elif isinstance(expr, Variable):
            return self._evaluate_variable(expr)
        elif isinstance(expr, PropertyAccess):
            return self._evaluate_property_access(expr)
        elif isinstance(expr, PropertyLookup):
            return self._evaluate_property_lookup(expr)
        elif isinstance(expr, BinaryOp):
            return self._evaluate_binary_op(expr)
        elif isinstance(expr, UnaryOp):
            return self._evaluate_unary_op(expr)
        elif isinstance(expr, LabelPredicate):
            return self._evaluate_label_predicate(expr)
        elif isinstance(expr, CaseExpression):
            return self._evaluate_case_expression(expr)
        elif isinstance(expr, ListLiteral):
            return self._evaluate_list_literal(expr)
        elif isinstance(expr, ListComprehension):
            return self._evaluate_list_comprehension(expr)
        elif isinstance(expr, ListIndex):
            return self._evaluate_list_index(expr)
        elif isinstance(expr, ListSlice):
            return self._evaluate_list_slice(expr)
        elif isinstance(expr, ListPredicate):
            return self._evaluate_list_predicate(expr)
        elif isinstance(expr, PatternComprehension):
            return self._evaluate_pattern_comprehension(expr)
        elif isinstance(expr, FunctionCallExpression):
            return self._evaluate_function_call(expr)
        elif isinstance(expr, MapLiteral):
            return self._evaluate_map_literal(expr)
        elif isinstance(expr, ReduceExpression):
            return self._evaluate_reduce_expression(expr)
        else:
            raise CypherExecutionError(f"Unknown expression type: {type(expr)}")

    def _evaluate_literal(self, expr: Literal) -> Any:
        """Evaluate a literal value."""
        return expr.value

    def _evaluate_parameter(self, expr: Parameter) -> Any:
        """Resolve a $name parameter reference from the supplied query parameters."""
        if expr.name not in self.parameters:
            raise CypherExecutionError(f"Parameter '${expr.name}' was not provided")
        return self.parameters[expr.name]

    def _evaluate_variable(self, expr: Variable) -> Any:
        """Evaluate a variable reference."""
        if expr.name not in self.context:
            raise CypherExecutionError(f"Variable '{expr.name}' not found in context")
        return self.context[expr.name]

    def _evaluate_property_access(self, expr: PropertyAccess) -> Any:
        """Evaluate property access (e.g., n.age).

        Args:
            expr: PropertyAccess node

        Returns:
            The property value

        Raises:
            CypherExecutionError: If variable or property doesn't exist
        """
        if expr.variable not in self.context:
            raise CypherExecutionError(f"Variable '{expr.variable}' not found in context")

        obj = self.context[expr.variable]

        # Handle Node and Relationship objects
        if hasattr(obj, 'properties'):
            return obj.properties.get(expr.property)
        if isinstance(obj, dict):
            return obj.get(expr.property)
        if isinstance(obj, list):
            values = []
            for item in obj:
                if hasattr(item, 'properties'):
                    values.append(item.properties.get(expr.property))
                elif isinstance(item, dict):
                    values.append(item.get(expr.property))
                else:
                    values.append(None)
            return values
        else:
            raise CypherExecutionError(
                f"Variable '{expr.variable}' does not have properties"
            )

    def _evaluate_property_lookup(self, expr: PropertyLookup) -> Any:
        """Evaluate property lookup on any expression."""
        base_value = self.evaluate(expr.base_expr)
        return self._lookup_property(base_value, expr.property)

    def _lookup_property(self, value: Any, prop: str) -> Any:
        """Resolve a property from supported value types."""
        if value is None:
            return None
        if hasattr(value, 'properties'):
            return value.properties.get(prop)
        if isinstance(value, dict):
            return value.get(prop)
        if isinstance(value, list):
            return [self._lookup_property(item, prop) for item in value]
        return getattr(value, prop, None)

    def _evaluate_binary_op(self, expr: BinaryOp) -> Any:
        """Evaluate binary operation.

        Supports: =, !=, <, >, <=, >=, AND, OR

        Args:
            expr: BinaryOp node

        Returns:
            The result of the operation
        """
        operator = expr.operator

        # Logical operators (short-circuit evaluation)
        if operator == 'AND':
            left = self.evaluate(expr.left)
            if left is False:
                return False
            if left is True:
                return self._to_tristate(self.evaluate(expr.right))
            right = self.evaluate(expr.right)
            if right is False:
                return False
            return None

        elif operator == 'OR':
            left = self.evaluate(expr.left)
            if left is True:
                return True
            if left is False:
                return self._to_tristate(self.evaluate(expr.right))
            right = self.evaluate(expr.right)
            if right is True:
                return True
            return None

        if operator == 'IN':
            left = self.evaluate(expr.left)
            right = self.evaluate(expr.right)
            if left is None or right is None:
                return None
            try:
                if left in right:
                    return True
                if any(item is None for item in right):
                    return None
                return False
            except TypeError:
                return None

        if operator == '+':
            left = self.evaluate(expr.left)
            right = self.evaluate(expr.right)
            if left is None or right is None:
                return None
            if isinstance(left, list) and isinstance(right, list):
                return left + right
            if isinstance(left, (date, datetime)) and isinstance(right, timedelta):
                return left + right
            if isinstance(right, (date, datetime)) and isinstance(left, timedelta):
                return right + left
            if isinstance(left, time) and isinstance(right, timedelta):
                return self._add_time_delta(left, right)
            if isinstance(right, time) and isinstance(left, timedelta):
                return self._add_time_delta(right, left)
            if isinstance(left, timedelta) and isinstance(right, timedelta):
                return left + right
            return left + right
        if operator == '-':
            left = self.evaluate(expr.left)
            right = self.evaluate(expr.right)
            if left is None or right is None:
                return None
            if isinstance(left, datetime) and isinstance(right, datetime):
                return left - right
            if isinstance(left, date) and isinstance(right, date):
                return left - right
            if isinstance(left, (date, datetime)) and isinstance(right, timedelta):
                return left - right
            if isinstance(left, time) and isinstance(right, timedelta):
                return self._add_time_delta(left, -right)
            if isinstance(left, timedelta) and isinstance(right, timedelta):
                return left - right
            return left - right
        if operator in ('*', '/', '%', '^'):
            left = self.evaluate(expr.left)
            right = self.evaluate(expr.right)
            if left is None or right is None:
                return None
            if operator == '*':
                return left * right
            if operator == '/':
                if right == 0:
                    raise CypherExecutionError("/ by zero")
                # Integer division when both operands are ints (Cypher semantics).
                if isinstance(left, int) and isinstance(right, int):
                    return int(left / right)
                return left / right
            if operator == '%':
                if right == 0:
                    raise CypherExecutionError("% by zero")
                return left % right
            if operator == '^':
                return float(left ** right)

        # Comparison operators (evaluate both sides first)
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)

        if operator == '=':
            return self._compare_values(left, right, '=')
        elif operator == '!=':
            return self._compare_values(left, right, '!=')
        elif operator == '<':
            return self._compare_values(left, right, '<')
        elif operator == '>':
            return self._compare_values(left, right, '>')
        elif operator == '<=':
            return self._compare_values(left, right, '<=')
        elif operator == '>=':
            return self._compare_values(left, right, '>=')
        else:
            raise CypherExecutionError(f"Unknown binary operator: {operator}")

    def _evaluate_unary_op(self, expr: UnaryOp) -> Any:
        """Evaluate unary operation.

        Supports: NOT, unary +/-

        Args:
            expr: UnaryOp node

        Returns:
            The result of the operation
        """
        if expr.operator == 'NOT':
            operand = self.evaluate(expr.operand)
            if operand is None:
                return None
            return not bool(operand)
        if expr.operator == '+':
            operand = self.evaluate(expr.operand)
            if operand is None:
                return None
            return operand
        if expr.operator == '-':
            operand = self.evaluate(expr.operand)
            if operand is None:
                return None
            return -operand
        if expr.operator == 'IS NULL':
            return self.evaluate(expr.operand) is None
        if expr.operator == 'IS NOT NULL':
            return self.evaluate(expr.operand) is not None
        else:
            raise CypherExecutionError(f"Unknown unary operator: {expr.operator}")

    def _evaluate_label_predicate(self, expr: LabelPredicate) -> Any:
        """Evaluate `n:Label` (or `n:A:B`): true iff the operand node carries
        all of the given labels. Null operand propagates null."""
        value = self.evaluate(expr.operand)
        if value is None:
            return None
        if hasattr(value, 'labels'):
            node_labels = value.labels
        elif isinstance(value, dict) and 'labels' in value:
            node_labels = value['labels']
        else:
            raise CypherExecutionError("label predicate (n:Label) expects a node")
        return all(label in node_labels for label in expr.labels)

    def _compare_values(self, left: Any, right: Any, operator: str) -> Any:
        """Compare two values with proper type handling.

        Args:
            left: Left operand
            right: Right operand
            operator: Comparison operator (<, >, <=, >=)

        Returns:
            Comparison result

        Raises:
            CypherExecutionError: If values cannot be compared
        """
        if left is None or right is None:
            return None
        # Handle list comparisons (e.g., r.prop where r is a list of relationships)
        if isinstance(left, list) and not isinstance(right, list):
            if operator == '=':
                return right in left
            if operator == '!=':
                return right not in left
        if isinstance(right, list) and not isinstance(left, list):
            if operator == '=':
                return left in right
            if operator == '!=':
                return left not in right
        if isinstance(left, list) and isinstance(right, list):
            if operator == '=':
                return left == right
            if operator == '!=':
                return left != right

        # Attempt comparison
        try:
            if operator == '<':
                return left < right
            elif operator == '>':
                return left > right
            elif operator == '<=':
                return left <= right
            elif operator == '>=':
                return left >= right
            elif operator == '=':
                return left == right
            elif operator == '!=':
                return left != right
            else:
                raise CypherExecutionError(f"Unknown comparison operator: {operator}")
        except TypeError:
            return None

    def _to_tristate(self, value: Any) -> Any:
        """Normalize a value into True/False/None."""
        if value is None:
            return None
        return bool(value)

    def _evaluate_list_literal(self, expr: ListLiteral) -> Any:
        """Evaluate a list literal."""
        return [self.evaluate(item) for item in expr.items]

    def _evaluate_list_comprehension(self, expr: ListComprehension) -> Any:
        """Evaluate a list comprehension."""
        source = self.evaluate(expr.list_expr)
        if source is None:
            return []
        if not isinstance(source, (list, tuple)):
            raise CypherExecutionError("List comprehension source must be a list")

        results = []
        had_var = expr.variable in self.context
        old_value = self.context.get(expr.variable)

        for item in source:
            self.context[expr.variable] = item
            if expr.where_expr is not None:
                try:
                    if not self.evaluate(expr.where_expr):
                        continue
                except CypherExecutionError:
                    continue
            results.append(self.evaluate(expr.projection))

        if had_var:
            self.context[expr.variable] = old_value
        else:
            self.context.pop(expr.variable, None)

        return results

    def _evaluate_list_index(self, expr: ListIndex) -> Any:
        """Evaluate list indexing."""
        source = self.evaluate(expr.list_expr)
        index = self.evaluate(expr.index_expr)
        if isinstance(source, dict):
            return source.get(index)
        if not isinstance(source, (list, tuple)):
            raise CypherExecutionError("List indexing source must be a list")
        if not isinstance(index, int):
            raise CypherExecutionError("List index must be an integer")
        try:
            return source[index]
        except IndexError:
            return None

    def _evaluate_list_slice(self, expr: ListSlice) -> Any:
        """Evaluate list slicing."""
        source = self.evaluate(expr.list_expr)
        if not isinstance(source, (list, tuple)):
            raise CypherExecutionError("List slicing source must be a list")
        start = self.evaluate(expr.start_expr) if expr.start_expr is not None else None
        end = self.evaluate(expr.end_expr) if expr.end_expr is not None else None
        if start is not None and not isinstance(start, int):
            raise CypherExecutionError("Slice start must be an integer")
        if end is not None and not isinstance(end, int):
            raise CypherExecutionError("Slice end must be an integer")
        return list(source)[start:end]

    def _evaluate_list_predicate(self, expr: ListPredicate) -> Any:
        """Evaluate list predicates ANY/ALL/NONE/SINGLE."""
        source = self.evaluate(expr.list_expr)
        if source is None:
            source = []
        if not isinstance(source, (list, tuple)):
            raise CypherExecutionError("List predicate source must be a list")

        had_var = expr.variable in self.context
        old_value = self.context.get(expr.variable)

        true_count = 0
        null_count = 0
        for item in source:
            self.context[expr.variable] = item
            if expr.where_expr is None:
                if item is None:
                    condition = None
                else:
                    condition = bool(item)
            else:
                condition = self._to_tristate(self.evaluate(expr.where_expr))

            if condition is True:
                true_count += 1
                if expr.predicate == 'ANY':
                    break
            elif condition is None:
                null_count += 1

        if had_var:
            self.context[expr.variable] = old_value
        else:
            self.context.pop(expr.variable, None)

        if expr.predicate == 'ANY':
            if true_count > 0:
                return True
            if null_count > 0:
                return None
            return False
        if expr.predicate == 'ALL':
            if true_count + null_count < len(source):
                return False
            if null_count > 0:
                return None
            return True
        if expr.predicate == 'NONE':
            if true_count > 0:
                return False
            if null_count > 0:
                return None
            return True
        if expr.predicate == 'SINGLE':
            if true_count > 1:
                return False
            if true_count == 1 and null_count == 0:
                return True
            if true_count == 0 and null_count == 0:
                return False
            return None

        raise CypherExecutionError(f"Unknown predicate: {expr.predicate}")

    def _evaluate_pattern_comprehension(self, expr: PatternComprehension) -> Any:
        """Evaluate a pattern comprehension using the provided matcher."""
        if self.pattern_matcher is None:
            raise CypherExecutionError("Pattern comprehensions require a query context")
        return self.pattern_matcher(expr, self.context)

    def _evaluate_map_literal(self, expr: MapLiteral) -> Any:
        """Evaluate a map literal."""
        return {key: self.evaluate(value) for key, value in expr.items.items()}

    def _evaluate_reduce_expression(self, expr: ReduceExpression) -> Any:
        """Evaluate reduce expression."""
        source = self.evaluate(expr.list_expr)
        if source is None:
            return None
        if not isinstance(source, (list, tuple)):
            raise CypherExecutionError("reduce() source must be a list")

        acc_value = self.evaluate(expr.init_expr)

        had_var = expr.variable in self.context
        old_var = self.context.get(expr.variable)
        had_acc = expr.accumulator in self.context
        old_acc = self.context.get(expr.accumulator)

        for item in source:
            self.context[expr.variable] = item
            self.context[expr.accumulator] = acc_value
            acc_value = self.evaluate(expr.expression)

        if had_var:
            self.context[expr.variable] = old_var
        else:
            self.context.pop(expr.variable, None)
        if had_acc:
            self.context[expr.accumulator] = old_acc
        else:
            self.context.pop(expr.accumulator, None)

        return acc_value

    def _evaluate_function_call(self, expr: FunctionCallExpression) -> Any:
        """Evaluate list-related function calls."""
        name = expr.name.lower()
        args = [self.evaluate(arg) for arg in expr.arguments]

        if name in {'filter', 'extract', 'reduce'}:
            raise CypherExecutionError(f"{name}() is parsed as a list expression, not a function call")

        if name in {'tointeger', 'tofloat', 'tostring', 'toboolean'}:
            if len(args) != 1:
                raise CypherExecutionError(f"{name}() expects 1 argument")
            return self._evaluate_cast(name, args[0])

        if name in {'tointegerlist', 'tofloatlist', 'tostringlist', 'tobooleanlist'}:
            return self._evaluate_list_cast(name, args)

        if name == 'length':
            return self._evaluate_length(args)

        if name in {'toupper', 'tolower', 'trim', 'ltrim', 'rtrim', 'split', 'substring',
                    'matches', 'regex', 'replace', 'left', 'right'}:
            return self._evaluate_string_function(name, args)

        if name in {'abs', 'ceil', 'floor', 'round', 'sqrt', 'sign', 'exp', 'log',
                    'log10', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
                    'cot', 'degrees', 'radians', 'haversin', 'e', 'pi', 'rand'}:
            return self._evaluate_math_function(name, args)

        if name in {'startnode', 'endnode'}:
            return self._evaluate_endpoint_function(name, args)

        if name in {'deaccent', 'strip_html', 'strip_emoji', 'snake_case'}:
            return self._evaluate_text_cleanup_function(name, args)

        if name in {'levenshtein', 'jaccard'}:
            return self._evaluate_text_similarity_function(name, args)

        if name in {'similar', 'vector_score'}:
            return self._evaluate_vector_function(name, args)

        if name == 'point':
            if len(args) != 1:
                raise CypherExecutionError("point() expects 1 argument")
            return self._evaluate_point(args[0])

        if name == 'distance':
            if len(args) != 2:
                raise CypherExecutionError("distance() expects 2 arguments")
            return self._evaluate_distance(args[0], args[1])

        if name == 'coalesce':
            if not args:
                raise CypherExecutionError("coalesce() expects at least 1 argument")
            for value in args:
                if value is not None:
                    return value
            return None

        if name == 'exists':
            # Property form: exists(n.prop) is true iff the property is present
            # (a missing property evaluates to None), matching `n.prop IS NOT NULL`.
            if len(args) != 1:
                raise CypherExecutionError("exists() expects 1 argument")
            return args[0] is not None

        if name in {'nodes', 'relationships'}:
            if len(args) != 1:
                raise CypherExecutionError(f"{name}() expects 1 argument")
            value = args[0]
            if not isinstance(value, Path):
                raise CypherExecutionError(f"{name}() expects a path argument")
            return value.nodes if name == 'nodes' else value.relationships

        if name in {'keys', 'values'}:
            if len(args) != 1:
                raise CypherExecutionError(f"{name}() expects 1 argument")
            value = args[0]
            if value is None:
                return None
            if isinstance(value, dict):
                return list(value.keys()) if name == 'keys' else list(value.values())
            if hasattr(value, 'properties'):
                return list(value.properties.keys()) if name == 'keys' else list(value.properties.values())
            raise CypherExecutionError(f"{name}() expects a map, node, or relationship")

        if name == 'labels':
            if len(args) != 1:
                raise CypherExecutionError("labels() expects 1 argument")
            value = args[0]
            if value is None:
                return None
            if hasattr(value, 'labels'):
                return list(value.labels)
            if isinstance(value, dict) and 'labels' in value:
                return list(value['labels'])
            raise CypherExecutionError("labels() expects a node")

        if name == 'type':
            if len(args) != 1:
                raise CypherExecutionError("type() expects 1 argument")
            value = args[0]
            if value is None:
                return None
            if hasattr(value, 'type'):
                return value.type
            if isinstance(value, dict) and 'type' in value:
                return value['type']
            raise CypherExecutionError("type() expects a relationship")

        if name == 'properties':
            if len(args) != 1:
                raise CypherExecutionError("properties() expects 1 argument")
            value = args[0]
            if value is None:
                return None
            if hasattr(value, 'properties'):
                return dict(value.properties)
            if isinstance(value, dict):
                # properties(map) returns the map itself.
                return dict(value)
            raise CypherExecutionError("properties() expects a map, node, or relationship")

        if name in {'id', 'elementid'}:
            if len(args) != 1:
                raise CypherExecutionError(f"{name}() expects 1 argument")
            value = args[0]
            if value is None:
                return None
            if hasattr(value, 'id'):
                identity = value.id
            elif isinstance(value, dict) and 'id' in value:
                identity = value['id']
            else:
                raise CypherExecutionError(f"{name}() expects a node or relationship")
            # elementId() yields a string identifier; id() yields the integer.
            return str(identity) if name == 'elementid' else identity

        if name == 'apoc.text.join':
            if len(args) != 2:
                raise CypherExecutionError("apoc.text.join() expects 2 arguments")
            values, delim = args
            if values is None or delim is None:
                return None
            if not isinstance(values, (list, tuple)):
                raise CypherExecutionError("apoc.text.join() expects a list")
            if not isinstance(delim, str):
                raise CypherExecutionError("apoc.text.join() expects a string delimiter")
            if any(item is None for item in values):
                return None
            return delim.join(str(item) for item in values)

        if name == 'apoc.text.split':
            if len(args) != 2:
                raise CypherExecutionError("apoc.text.split() expects 2 arguments")
            value, pattern = args
            if value is None or pattern is None:
                return None
            if not isinstance(value, str) or not isinstance(pattern, str):
                raise CypherExecutionError("apoc.text.split() expects string arguments")
            import re
            try:
                return re.split(pattern, value)
            except re.error as exc:
                raise CypherExecutionError(f"apoc.text.split() invalid regex: {exc}") from exc

        if name == 'apoc.text.replace':
            if len(args) != 3:
                raise CypherExecutionError("apoc.text.replace() expects 3 arguments")
            value, pattern, replacement = args
            if value is None or pattern is None or replacement is None:
                return None
            if not isinstance(value, str) or not isinstance(pattern, str) or not isinstance(replacement, str):
                raise CypherExecutionError("apoc.text.replace() expects string arguments")
            import re
            try:
                return re.sub(pattern, replacement, value)
            except re.error as exc:
                raise CypherExecutionError(f"apoc.text.replace() invalid regex: {exc}") from exc

        if name == 'apoc.map.merge':
            if len(args) != 2:
                raise CypherExecutionError("apoc.map.merge() expects 2 arguments")
            left, right = args
            if left is None or right is None:
                return None
            if not isinstance(left, dict) or not isinstance(right, dict):
                raise CypherExecutionError("apoc.map.merge() expects map arguments")
            merged = dict(left)
            merged.update(right)
            return merged

        if name == 'apoc.convert.tomap':
            if len(args) != 1:
                raise CypherExecutionError("apoc.convert.toMap() expects 1 argument")
            value = args[0]
            if value is None:
                return None
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, (list, tuple)):
                mapped: dict[Any, Any] = {}
                for item in value:
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        raise CypherExecutionError("apoc.convert.toMap() expects pairs")
                    mapped[item[0]] = item[1]
                return mapped
            if hasattr(value, "properties"):
                return dict(value.properties)
            raise CypherExecutionError("apoc.convert.toMap() expects a map or list of pairs")

        if name == 'apoc.map.clean':
            if len(args) != 3:
                raise CypherExecutionError("apoc.map.clean() expects 3 arguments")
            data, remove_keys, remove_values = args
            if data is None:
                return None
            if not isinstance(data, dict):
                raise CypherExecutionError("apoc.map.clean() expects a map")
            if remove_keys is None:
                remove_keys = []
            if remove_values is None:
                remove_values = []
            if not isinstance(remove_keys, (list, tuple)):
                raise CypherExecutionError("apoc.map.clean() expects a list of keys")
            if not isinstance(remove_values, (list, tuple)):
                raise CypherExecutionError("apoc.map.clean() expects a list of values")
            cleaned: dict[Any, Any] = {}
            for key, value in data.items():
                if key in remove_keys:
                    continue
                if value in remove_values:
                    continue
                cleaned[key] = value
            return cleaned

        if name == 'apoc.map.frompairs':
            if len(args) != 1:
                raise CypherExecutionError("apoc.map.fromPairs() expects 1 argument")
            pairs = args[0]
            if pairs is None:
                return None
            if not isinstance(pairs, (list, tuple)):
                raise CypherExecutionError("apoc.map.fromPairs() expects a list of pairs")
            mapped: dict[Any, Any] = {}
            for item in pairs:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise CypherExecutionError("apoc.map.fromPairs() expects pairs")
                mapped[item[0]] = item[1]
            return mapped

        if name == 'apoc.map.removekey':
            if len(args) != 2:
                raise CypherExecutionError("apoc.map.removeKey() expects 2 arguments")
            data, key = args
            if data is None or key is None:
                return None
            if not isinstance(data, dict):
                raise CypherExecutionError("apoc.map.removeKey() expects a map")
            updated = dict(data)
            updated.pop(key, None)
            return updated

        if name == 'apoc.util.compress':
            if len(args) not in (1, 2):
                raise CypherExecutionError("apoc.util.compress() expects 1 or 2 arguments")
            value = args[0]
            options = args[1] if len(args) == 2 else None
            if value is None:
                return None
            if not isinstance(value, (str, bytes)):
                raise CypherExecutionError("apoc.util.compress() expects a string or bytes")
            if options is not None and not isinstance(options, dict):
                raise CypherExecutionError("apoc.util.compress() options must be a map")
            compression = (options or {}).get("compression", "DEFLATE")
            if not isinstance(compression, str):
                raise CypherExecutionError("apoc.util.compress() compression must be a string")
            compression = compression.upper()
            data = value.encode("utf-8") if isinstance(value, str) else value
            if compression == "DEFLATE":
                import zlib

                return zlib.compress(data)
            if compression in ("GZIP", "GZ"):
                import gzip

                return gzip.compress(data)
            if compression in ("BZ2", "BZIP2"):
                import bz2

                return bz2.compress(data)
            if compression in ("XZ", "LZMA"):
                import lzma

                return lzma.compress(data)
            raise CypherExecutionError(f"Unsupported compression type: {compression}")

        if name == 'apoc.map.get':
            if len(args) not in (2, 3):
                raise CypherExecutionError("apoc.map.get() expects 2 or 3 arguments")
            data = args[0]
            key = args[1]
            default = args[2] if len(args) == 3 else None
            if data is None or key is None:
                return default
            if not isinstance(data, dict):
                raise CypherExecutionError("apoc.map.get() expects a map")
            return data.get(key, default)

        if name == 'apoc.coll.contains':
            if len(args) != 2:
                raise CypherExecutionError("apoc.coll.contains() expects 2 arguments")
            values, target = args
            if values is None or target is None:
                return None
            if not isinstance(values, (list, tuple)):
                raise CypherExecutionError("apoc.coll.contains() expects a list")
            if target in values:
                return True
            if any(item is None for item in values):
                return None
            return False

        if name == 'apoc.coll.toset':
            if len(args) != 1:
                raise CypherExecutionError("apoc.coll.toSet() expects 1 argument")
            values = args[0]
            if values is None:
                return None
            if not isinstance(values, (list, tuple)):
                raise CypherExecutionError("apoc.coll.toSet() expects a list")
            return self._unique_list(values)

        if name in {'date', 'time', 'localtime', 'datetime', 'localdatetime', 'duration'}:
            if len(args) > 1:
                raise CypherExecutionError(f"{name}() expects 0 or 1 arguments")
            value = args[0] if args else None
            return self._evaluate_temporal_function(name, value)

        if name in {'date.truncate', 'datetime.truncate', 'localdatetime.truncate', 'time.truncate'}:
            if len(args) != 2:
                raise CypherExecutionError(f"{name}() expects 2 arguments")
            return self._evaluate_temporal_truncate(name, args[0], args[1])

        if name == 'duration.between':
            if len(args) != 2:
                raise CypherExecutionError("duration.between() expects 2 arguments")
            return self._evaluate_duration_between(args[0], args[1])

        if name == 'size':
            if len(args) != 1:
                raise CypherExecutionError("size() expects 1 argument")
            return len(args[0]) if args[0] is not None else None
        if name == 'head':
            if len(args) != 1:
                raise CypherExecutionError("head() expects 1 argument")
            return args[0][0] if args[0] else None
        if name == 'tail':
            if len(args) != 1:
                raise CypherExecutionError("tail() expects 1 argument")
            return args[0][1:] if args[0] else []
        if name == 'last':
            if len(args) != 1:
                raise CypherExecutionError("last() expects 1 argument")
            return args[0][-1] if args[0] else None
        if name == 'reverse':
            if len(args) != 1:
                raise CypherExecutionError("reverse() expects 1 argument")
            value = args[0]
            if value is None:
                return None
            # reverse() preserves the input type: string -> string, list -> list.
            if isinstance(value, str):
                return value[::-1]
            return list(reversed(value))
        if name == 'range':
            if len(args) not in (2, 3):
                raise CypherExecutionError("range() expects 2 or 3 arguments")
            start = args[0]
            end = args[1]
            step = args[2] if len(args) == 3 else 1
            if step == 0:
                raise CypherExecutionError("range() step cannot be 0")
            if step > 0:
                return list(range(start, end + 1, step))
            return list(range(start, end - 1, step))

        raise CypherExecutionError(f"Unknown function: {expr.name}")

    def _evaluate_string_function(self, name: str, args: list[Any]) -> Any:
        """Evaluate string-related functions."""
        if name in {'toupper', 'tolower', 'trim', 'ltrim', 'rtrim'}:
            if len(args) != 1:
                raise CypherExecutionError(f"{name}() expects 1 argument")
            value = args[0]
            if value is None:
                return None
            if not isinstance(value, str):
                raise CypherExecutionError(f"{name}() expects a string")
            if name == 'toupper':
                return value.upper()
            if name == 'tolower':
                return value.lower()
            if name == 'ltrim':
                return value.lstrip()
            if name == 'rtrim':
                return value.rstrip()
            return value.strip()

        if name == 'replace':
            if len(args) != 3:
                raise CypherExecutionError("replace() expects 3 arguments")
            value, search, replacement = args
            if value is None or search is None or replacement is None:
                return None
            if not all(isinstance(a, str) for a in (value, search, replacement)):
                raise CypherExecutionError("replace() expects string arguments")
            return value.replace(search, replacement)

        if name in {'left', 'right'}:
            if len(args) != 2:
                raise CypherExecutionError(f"{name}() expects 2 arguments")
            value, length = args
            if value is None or length is None:
                return None
            if not isinstance(value, str):
                raise CypherExecutionError(f"{name}() expects a string")
            if not isinstance(length, int) or isinstance(length, bool):
                raise CypherExecutionError(f"{name}() length must be an integer")
            if length < 0:
                raise CypherExecutionError(f"{name}() length must be >= 0")
            return value[:length] if name == 'left' else (value[-length:] if length else '')

        if name == 'split':
            if len(args) != 2:
                raise CypherExecutionError("split() expects 2 arguments")
            value, delim = args
            if value is None or delim is None:
                return None
            if not isinstance(value, str) or not isinstance(delim, str):
                raise CypherExecutionError("split() expects string arguments")
            return value.split(delim)

        if name == 'substring':
            if len(args) not in (2, 3):
                raise CypherExecutionError("substring() expects 2 or 3 arguments")
            value = args[0]
            start = args[1]
            length = args[2] if len(args) == 3 else None
            if value is None or start is None:
                return None
            if not isinstance(value, str):
                raise CypherExecutionError("substring() expects a string")
            if not isinstance(start, int):
                raise CypherExecutionError("substring() start must be an integer")
            if start < 0:
                raise CypherExecutionError("substring() start must be >= 0")
            if length is not None and not isinstance(length, int):
                raise CypherExecutionError("substring() length must be an integer")
            if length is not None and length < 0:
                raise CypherExecutionError("substring() length must be >= 0")
            if length is None:
                return value[start:]
            return value[start:start + length]

        if name in {'matches', 'regex'}:
            if len(args) != 2:
                raise CypherExecutionError(f"{name}() expects 2 arguments")
            value, pattern = args
            if value is None or pattern is None:
                return None
            if not isinstance(value, str) or not isinstance(pattern, str):
                raise CypherExecutionError(f"{name}() expects string arguments")
            import re
            try:
                return re.search(pattern, value) is not None
            except re.error as exc:
                raise CypherExecutionError(f"{name}() invalid regex: {exc}") from exc

        raise CypherExecutionError(f"Unknown string function: {name}")

    def _evaluate_math_function(self, name: str, args: list[Any]) -> Any:
        """Evaluate numeric/math functions."""
        import math

        # Constants take no arguments.
        if name in {'e', 'pi', 'rand'}:
            if args:
                raise CypherExecutionError(f"{name}() expects no arguments")
            if name == 'e':
                return math.e
            if name == 'pi':
                return math.pi
            import random
            return random.random()

        # atan2(y, x) takes two numeric arguments.
        if name == 'atan2':
            if len(args) != 2:
                raise CypherExecutionError("atan2() expects 2 arguments")
            y, x = args
            if y is None or x is None:
                return None
            for a in (y, x):
                if isinstance(a, bool) or not isinstance(a, (int, float)):
                    raise CypherExecutionError("atan2() expects numbers")
            return math.atan2(y, x)

        if len(args) != 1:
            raise CypherExecutionError(f"{name}() expects 1 argument")
        value = args[0]
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CypherExecutionError(f"{name}() expects a number")

        if name == 'abs':
            return abs(value)
        if name == 'ceil':
            return float(math.ceil(value))
        if name == 'floor':
            return float(math.floor(value))
        if name == 'round':
            # Cypher rounds halves away from zero (unlike Python's banker's rounding).
            return float(math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5))
        if name == 'sqrt':
            if value < 0:
                raise CypherExecutionError("sqrt() of a negative number")
            return math.sqrt(value)
        if name == 'sign':
            return (value > 0) - (value < 0)
        if name == 'exp':
            return math.exp(value)
        if name == 'log':
            if value <= 0:
                raise CypherExecutionError("log() requires a positive number")
            return math.log(value)
        if name == 'log10':
            if value <= 0:
                raise CypherExecutionError("log10() requires a positive number")
            return math.log10(value)
        if name == 'sin':
            return math.sin(value)
        if name == 'cos':
            return math.cos(value)
        if name == 'tan':
            return math.tan(value)
        if name == 'cot':
            return 1.0 / math.tan(value)
        if name in {'asin', 'acos'}:
            if value < -1 or value > 1:
                raise CypherExecutionError(f"{name}() requires an argument in [-1, 1]")
            return math.asin(value) if name == 'asin' else math.acos(value)
        if name == 'atan':
            return math.atan(value)
        if name == 'degrees':
            return math.degrees(value)
        if name == 'radians':
            return math.radians(value)
        if name == 'haversin':
            return (1.0 - math.cos(value)) / 2.0
        raise CypherExecutionError(f"Unknown math function: {name}")

    def _evaluate_endpoint_function(self, name: str, args: list[Any]) -> Any:
        """Evaluate startNode()/endNode() over a relationship."""
        if len(args) != 1:
            raise CypherExecutionError(f"{name}() expects 1 argument")
        rel = args[0]
        if rel is None:
            return None
        node_id = getattr(rel, 'source_id' if name == 'startnode' else 'target_id', None)
        if node_id is None:
            raise CypherExecutionError(f"{name}() expects a relationship")
        if self.node_resolver is None:
            raise CypherExecutionError(f"{name}() cannot resolve nodes in this context")
        return self.node_resolver(node_id)

    def _evaluate_text_cleanup_function(self, name: str, args: list[Any]) -> Any:
        """Evaluate text cleanup functions."""
        if len(args) != 1:
            raise CypherExecutionError(f"{name}() expects 1 argument")
        value = args[0]
        if value is None:
            return None
        if not isinstance(value, str):
            raise CypherExecutionError(f"{name}() expects a string")
        if name == 'deaccent':
            return self._deaccent_text(value)
        if name == 'strip_html':
            return self._strip_html(value)
        if name == 'strip_emoji':
            return self._strip_emoji(value)
        if name == 'snake_case':
            return self._to_snake_case(value)
        raise CypherExecutionError(f"Unknown text function: {name}")

    def _evaluate_vector_function(self, name: str, args: list[Any]) -> Any:
        """Evaluate SIMILAR() / VECTOR_SCORE() against a node's stored embedding.

        Both take ``(node, query)`` plus an optional third argument: a map of
        options (``index``, ``min_score``) or, for ``SIMILAR()`` only, a bare
        number used as ``min_score``. ``VECTOR_SCORE()`` returns the raw score;
        ``SIMILAR()`` compares it against ``min_score``.

        These are *predicates*, not seeks: they score the node they are given
        and never consult the ANN index for neighbours. Filtering a bare
        ``MATCH (n)`` with them scans every node — seed the pattern with
        ``CALL db.vector.search(...)`` and use these to constrain the far end.
        """
        if not 2 <= len(args) <= 3:
            raise CypherExecutionError(f"{name}() expects 2 or 3 arguments")
        node, query = args[0], args[1]

        index = "default"
        min_score = None
        if len(args) == 3:
            options = args[2]
            if isinstance(options, dict):
                unknown = set(options) - {"index", "min_score"}
                if unknown:
                    raise CypherExecutionError(
                        f"{name}() got unknown options: {', '.join(sorted(unknown))}"
                    )
                index = options.get("index", index)
                if not isinstance(index, str):
                    raise CypherExecutionError(f"{name}() index option must be a string")
                min_score = options.get("min_score")
            elif isinstance(options, (int, float)) and not isinstance(options, bool):
                if name == 'vector_score':
                    raise CypherExecutionError(
                        "vector_score() third argument must be an options map"
                    )
                min_score = options
            else:
                raise CypherExecutionError(
                    f"{name}() third argument must be an options map"
                    + ("" if name == 'vector_score' else " or a number")
                )
            if min_score is not None and (
                isinstance(min_score, bool) or not isinstance(min_score, (int, float))
            ):
                raise CypherExecutionError(f"{name}() min_score must be a number")

        # NULL in, NULL out — an unmatched optional node is unknown, not dissimilar.
        if node is None or query is None:
            return None
        if not hasattr(node, 'id'):
            raise CypherExecutionError(f"{name}() expects a node as first argument")
        if not isinstance(query, (str, list)):
            raise CypherExecutionError(
                f"{name}() query must be a string or a list of floats"
            )
        if self.vector_scorer is None:
            raise CypherExecutionError(f"{name}() is not available in this context")

        score = self.vector_scorer(node, query, index)
        if name == 'vector_score':
            return score
        if score is None:
            # No embedding for this node: not similar, rather than unknown, so
            # that WHERE SIMILAR(...) drops it instead of propagating NULL.
            return False
        if min_score is None:
            min_score = self._default_min_score(name, index)
        return score >= min_score

    def _default_min_score(self, name: str, index: str) -> float:
        """Pick a default SIMILAR() threshold, refusing where none is meaningful."""
        metric = self.vector_metric(index) if self.vector_metric else "cosine"
        if metric == "l2":
            # l2 scores are negated squared distances (<= 0), so any positive
            # default would silently match nothing. Make the caller choose.
            raise CypherExecutionError(
                f"{name}() requires an explicit min_score on the '{index}' index "
                "because its 'l2' metric produces negative scores "
                f"(e.g. {name}(n, \"query\", -0.5))"
            )
        return DEFAULT_SIMILAR_MIN_SCORE

    def _evaluate_text_similarity_function(self, name: str, args: list[Any]) -> Any:
        """Evaluate text similarity functions."""
        if len(args) != 2:
            raise CypherExecutionError(f"{name}() expects 2 arguments")
        left, right = args
        if left is None or right is None:
            return None
        if not isinstance(left, str) or not isinstance(right, str):
            raise CypherExecutionError(f"{name}() expects string arguments")
        if name == 'levenshtein':
            return self._levenshtein_distance(left, right)
        if name == 'jaccard':
            return self._jaccard_similarity(left, right)
        raise CypherExecutionError(f"Unknown text similarity function: {name}")

    def _deaccent_text(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    def _strip_html(self, value: str) -> str:
        return re.sub(r"<[^>]*>", "", value)

    def _strip_emoji(self, value: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001F1E6-\U0001F1FF"
            "\U0001F300-\U0001F5FF"
            "\U0001F600-\U0001F64F"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FAFF"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", value)

    def _to_snake_case(self, value: str) -> str:
        """Convert text to snake_case for use as identifiers.

        Converts to ASCII, lowercases, replaces non-alphanumeric with underscores.
        """
        # Remove accents first
        text = self._deaccent_text(value)
        # Insert underscore before uppercase letters (for camelCase/PascalCase)
        text = re.sub(r'([a-z])([A-Z])', r'\1_\2', text)
        # Replace non-alphanumeric with underscores
        text = re.sub(r'[^a-zA-Z0-9]+', '_', text)
        # Lowercase
        text = text.lower()
        # Collapse multiple underscores
        text = re.sub(r'_+', '_', text)
        # Strip leading/trailing underscores
        return text.strip('_')

    def _levenshtein_distance(self, left: str, right: str) -> int:
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)
        if len(left) > len(right):
            left, right = right, left
        previous = list(range(len(left) + 1))
        for i, ch_right in enumerate(right, start=1):
            current = [i]
            for j, ch_left in enumerate(left, start=1):
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + (ch_left != ch_right)
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        return previous[-1]

    def _jaccard_similarity(self, left: str, right: str) -> float:
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens and not right_tokens:
            return 1.0
        union = left_tokens | right_tokens
        if not union:
            return 0.0
        return len(left_tokens & right_tokens) / len(union)

    def _unique_list(self, values: list[Any]) -> list[Any]:
        """Return a list of unique values preserving order."""
        unique = []
        for item in values:
            if item in unique:
                continue
            unique.append(item)
        return unique

    def _evaluate_point(self, value: Any) -> Point:
        """Evaluate point() from a map."""
        if not isinstance(value, dict):
            raise CypherExecutionError("point() expects a map argument")

        if 'longitude' in value or 'latitude' in value:
            if 'longitude' not in value or 'latitude' not in value:
                raise CypherExecutionError("point() requires longitude and latitude")
            return Point(
                longitude=float(value['longitude']),
                latitude=float(value['latitude']),
                height=float(value['height']) if value.get('height') is not None else None,
                srid=value.get('srid')
            )

        if 'x' not in value or 'y' not in value:
            raise CypherExecutionError("point() requires x and y for cartesian points")

        return Point(
            x=float(value['x']),
            y=float(value['y']),
            z=float(value['z']) if value.get('z') is not None else None,
            srid=value.get('srid')
        )

    def _evaluate_distance(self, left: Any, right: Any) -> float | None:
        """Evaluate distance between two points."""
        if left is None or right is None:
            return None
        if not isinstance(left, Point) or not isinstance(right, Point):
            raise CypherExecutionError("distance() expects point arguments")

        if left.longitude is not None and right.longitude is not None:
            return self._haversine_distance(left, right)
        if left.x is None or left.y is None or right.x is None or right.y is None:
            raise CypherExecutionError("distance() expects matching point types")

        dx = left.x - right.x
        dy = left.y - right.y
        dz = (left.z or 0.0) - (right.z or 0.0)
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def _haversine_distance(self, left: Point, right: Point) -> float:
        """Compute Haversine distance for WGS84 points in meters."""
        import math

        radius = 6371000.0
        lat1 = math.radians(left.latitude)
        lat2 = math.radians(right.latitude)
        dlat = lat2 - lat1
        dlon = math.radians(right.longitude - left.longitude)

        sin_dlat = math.sin(dlat / 2.0)
        sin_dlon = math.sin(dlon / 2.0)
        a = sin_dlat * sin_dlat + math.cos(lat1) * math.cos(lat2) * sin_dlon * sin_dlon
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return radius * c

    def _evaluate_cast(self, name: str, value: Any) -> Any:
        """Evaluate explicit casting functions."""
        if value is None:
            return None
        if name == 'tointeger':
            if isinstance(value, bool):
                return 1 if value else 0
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return int(float(value))
        if name == 'tofloat':
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value)
        if name == 'tostring':
            return str(value)
        if name == 'toboolean':
            if isinstance(value, bool):
                return value
            if isinstance(value, int):
                return value != 0
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered == 'true':
                    return True
                if lowered == 'false':
                    return False
                return None  # Cypher returns null for unparseable strings.

        raise CypherExecutionError(f"Cannot cast value using {name}()")

    def _evaluate_list_cast(self, name: str, args: list[Any]) -> Any:
        """Evaluate toIntegerList/toFloatList/toStringList/toBooleanList.

        Element-wise cast over a list; an element that cannot be converted
        becomes null (Cypher semantics), rather than raising.
        """
        if len(args) != 1:
            raise CypherExecutionError(f"{name}() expects 1 argument")
        value = args[0]
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise CypherExecutionError(f"{name}() expects a list")
        scalar = name[:-4]  # strip trailing 'list' -> tointeger/tofloat/...
        result = []
        for element in value:
            try:
                result.append(self._evaluate_cast(scalar, element))
            except (ValueError, TypeError, CypherExecutionError):
                result.append(None)
        return result

    def _evaluate_length(self, args: list[Any]) -> Any:
        """Evaluate length(): relationships in a path, or string/list length."""
        if len(args) != 1:
            raise CypherExecutionError("length() expects 1 argument")
        value = args[0]
        if value is None:
            return None
        if isinstance(value, Path):
            return len(value.relationships)
        if isinstance(value, (str, list, tuple)):
            return len(value)
        raise CypherExecutionError("length() expects a path, string, or list")

    def _evaluate_temporal_function(self, name: str, value: Any) -> Any:
        """Evaluate temporal function calls."""
        if value is None:
            if name == 'date':
                return date.today()
            if name in {'datetime', 'localdatetime'}:
                return datetime.now()
            if name in {'time', 'localtime'}:
                return datetime.now().time()
            if name == 'duration':
                return timedelta(0)
        if not isinstance(value, str):
            raise CypherExecutionError(f"{name}() expects an ISO-8601 string argument")

        value = value.strip()

        if name == 'date':
            return date.fromisoformat(value)
        if name in {'time', 'localtime'}:
            parsed = time.fromisoformat(value)
            if name == 'localtime' and parsed.tzinfo is not None:
                return parsed.replace(tzinfo=None)
            return parsed
        if name in {'datetime', 'localdatetime'}:
            iso_value = value.replace(' ', 'T', 1).replace('Z', '+00:00')
            parsed = datetime.fromisoformat(iso_value)
            if name == 'localdatetime' and parsed.tzinfo is not None:
                return parsed.replace(tzinfo=None)
            return parsed
        if name == 'duration':
            return self._parse_duration(value)

        raise CypherExecutionError(f"Unknown temporal function: {name}")

    def _evaluate_temporal_truncate(self, name: str, unit: Any, value: Any) -> Any:
        """Truncate temporal values to a unit."""
        if not isinstance(unit, str):
            raise CypherExecutionError("truncate() unit must be a string")
        unit = unit.lower()

        if name == 'date.truncate':
            if not isinstance(value, date) or isinstance(value, datetime):
                raise CypherExecutionError("date.truncate() expects a date")
            return self._truncate_date(unit, value)

        if name in {'datetime.truncate', 'localdatetime.truncate'}:
            if not isinstance(value, datetime):
                raise CypherExecutionError(f"{name}() expects a datetime")
            return self._truncate_datetime(unit, value)

        if name == 'time.truncate':
            if not isinstance(value, time):
                raise CypherExecutionError("time.truncate() expects a time")
            return self._truncate_time(unit, value)

        raise CypherExecutionError(f"Unknown temporal truncate: {name}")

    def _truncate_date(self, unit: str, value: date) -> date:
        """Truncate date to a unit."""
        if unit == 'year':
            return date(value.year, 1, 1)
        if unit == 'quarter':
            month = ((value.month - 1) // 3) * 3 + 1
            return date(value.year, month, 1)
        if unit == 'month':
            return date(value.year, value.month, 1)
        if unit == 'week':
            return value - timedelta(days=value.weekday())
        if unit == 'day':
            return value
        raise CypherExecutionError(f"Unsupported date.truncate unit: {unit}")

    def _truncate_datetime(self, unit: str, value: datetime) -> datetime:
        """Truncate datetime to a unit."""
        if unit in {'year', 'quarter', 'month', 'week', 'day'}:
            base = self._truncate_date(unit, value.date())
            return datetime.combine(base, time.min, tzinfo=value.tzinfo)
        if unit == 'hour':
            return value.replace(minute=0, second=0, microsecond=0)
        if unit == 'minute':
            return value.replace(second=0, microsecond=0)
        if unit == 'second':
            return value.replace(microsecond=0)
        raise CypherExecutionError(f"Unsupported datetime.truncate unit: {unit}")

    def _truncate_time(self, unit: str, value: time) -> time:
        """Truncate time to a unit."""
        if unit == 'hour':
            return value.replace(minute=0, second=0, microsecond=0)
        if unit == 'minute':
            return value.replace(second=0, microsecond=0)
        if unit == 'second':
            return value.replace(microsecond=0)
        raise CypherExecutionError(f"Unsupported time.truncate unit: {unit}")

    def _evaluate_duration_between(self, start: Any, end: Any) -> timedelta | None:
        """Evaluate duration.between(start, end)."""
        if start is None or end is None:
            return None

        if isinstance(start, date) and isinstance(end, date) and not isinstance(start, datetime) and not isinstance(end, datetime):
            return end - start
        if isinstance(start, datetime) and isinstance(end, datetime):
            if (start.tzinfo is None) != (end.tzinfo is None):
                return None
            return end - start
        if isinstance(start, time) and isinstance(end, time):
            base = date.min
            start_dt = datetime.combine(base, start)
            end_dt = datetime.combine(base, end)
            return end_dt - start_dt

        raise CypherExecutionError("duration.between() expects two dates, datetimes, or times")

    def _parse_duration(self, value: str) -> timedelta:
        """Parse a basic ISO-8601 duration string."""
        if not value.startswith('P'):
            raise CypherExecutionError("duration() expects an ISO-8601 duration string")

        import re

        pattern = re.compile(
            r"^P"
            r"(?:(?P<weeks>\d+(?:\.\d+)?)W)?"
            r"(?:(?P<days>\d+(?:\.\d+)?)D)?"
            r"(?:T"
            r"(?:(?P<hours>\d+(?:\.\d+)?)H)?"
            r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
            r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
            r")?$"
        )
        match = pattern.match(value)
        if not match:
            raise CypherExecutionError("duration() expects an ISO-8601 duration string")

        weeks = float(match.group('weeks') or 0)
        days = float(match.group('days') or 0) + weeks * 7
        hours = float(match.group('hours') or 0)
        minutes = float(match.group('minutes') or 0)
        seconds = float(match.group('seconds') or 0)

        return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    def _add_time_delta(self, base: time, delta: timedelta) -> time:
        """Add a timedelta to a time, wrapping within the day."""
        anchor = datetime.combine(date.min, base)
        return (anchor + delta).time()

    def _evaluate_case_expression(self, expr: CaseExpression) -> Any:
        """Evaluate CASE expression."""
        base_value = None
        if expr.base_expr is not None:
            base_value = self.evaluate(expr.base_expr)

        for when_clause in expr.whens:
            if expr.base_expr is None:
                condition = self.evaluate(when_clause.when_expr)
                if condition:
                    return self.evaluate(when_clause.then_expr)
            else:
                when_value = self.evaluate(when_clause.when_expr)
                if self._compare_values(base_value, when_value, '=') is True:
                    return self.evaluate(when_clause.then_expr)

        if expr.else_expr is not None:
            return self.evaluate(expr.else_expr)

        return None
