"""Tests for the Cypher expression evaluator."""

import pytest
from grafito.models import Node, Relationship, Path
from grafito.cypher.evaluator import ExpressionEvaluator
from grafito.cypher.ast_nodes import (
    Literal, PropertyAccess, BinaryOp, UnaryOp, CaseExpression, CaseWhen,
    Variable, ListLiteral, ListComprehension, ListIndex, ListSlice, ListPredicate,
    FunctionCallExpression, MapLiteral, ReduceExpression, PatternComprehension,
    Pattern, PatternElement, NodePattern, RelationshipPattern
)
from grafito.cypher.exceptions import CypherExecutionError


class TestEvaluatorLiterals:
    """Test evaluation of literal values."""

    def test_evaluate_integer(self):
        evaluator = ExpressionEvaluator({})
        expr = Literal(value=42)
        assert evaluator.evaluate(expr) == 42

    def test_evaluate_string(self):
        evaluator = ExpressionEvaluator({})
        expr = Literal(value="hello")
        assert evaluator.evaluate(expr) == "hello"

    def test_evaluate_boolean_true(self):
        evaluator = ExpressionEvaluator({})
        expr = Literal(value=True)
        assert evaluator.evaluate(expr) is True

    def test_evaluate_boolean_false(self):
        evaluator = ExpressionEvaluator({})
        expr = Literal(value=False)
        assert evaluator.evaluate(expr) is False

    def test_evaluate_null(self):
        evaluator = ExpressionEvaluator({})
        expr = Literal(value=None)
        assert evaluator.evaluate(expr) is None


class TestEvaluatorPropertyAccess:
    """Test evaluation of property access."""

    def test_property_access_simple(self):
        node = Node(id=1, labels=['Person'], properties={'name': 'Alice', 'age': 30})
        evaluator = ExpressionEvaluator({'n': node})
        expr = PropertyAccess(variable='n', property='name')
        assert evaluator.evaluate(expr) == 'Alice'

    def test_property_access_integer(self):
        node = Node(id=1, labels=['Person'], properties={'age': 30})
        evaluator = ExpressionEvaluator({'n': node})
        expr = PropertyAccess(variable='n', property='age')
        assert evaluator.evaluate(expr) == 30

    def test_property_access_missing_property(self):
        node = Node(id=1, labels=['Person'], properties={'name': 'Alice'})
        evaluator = ExpressionEvaluator({'n': node})
        expr = PropertyAccess(variable='n', property='age')
        assert evaluator.evaluate(expr) is None

    def test_property_access_missing_variable(self):
        evaluator = ExpressionEvaluator({})
        expr = PropertyAccess(variable='n', property='name')
        with pytest.raises(CypherExecutionError) as exc_info:
            evaluator.evaluate(expr)
        assert "not found in context" in str(exc_info.value)

    def test_property_access_on_relationship(self):
        rel = Relationship(id=1, source_id=1, target_id=2, type='KNOWS', properties={'since': 2020})
        evaluator = ExpressionEvaluator({'r': rel})
        expr = PropertyAccess(variable='r', property='since')
        assert evaluator.evaluate(expr) == 2020


class TestEvaluatorComparisons:
    """Test evaluation of comparison operations."""

    def test_equals_true(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(42), operator='=', right=Literal(42))
        assert evaluator.evaluate(expr) is True

    def test_equals_false(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(42), operator='=', right=Literal(24))
        assert evaluator.evaluate(expr) is False

    def test_not_equals(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(42), operator='!=', right=Literal(24))
        assert evaluator.evaluate(expr) is True

    def test_less_than(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(10), operator='<', right=Literal(20))
        assert evaluator.evaluate(expr) is True

    def test_greater_than(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(30), operator='>', right=Literal(20))
        assert evaluator.evaluate(expr) is True

    def test_less_than_or_equal(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(20), operator='<=', right=Literal(20))
        assert evaluator.evaluate(expr) is True

    def test_greater_than_or_equal(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(25), operator='>=', right=Literal(20))
        assert evaluator.evaluate(expr) is True

    def test_property_comparison(self):
        node = Node(id=1, labels=['Person'], properties={'age': 30})
        evaluator = ExpressionEvaluator({'n': node})
        expr = BinaryOp(
            left=PropertyAccess('n', 'age'),
            operator='>',
            right=Literal(25)
        )
        assert evaluator.evaluate(expr) is True

    def test_comparison_with_none(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(None), operator='>', right=Literal(10))
        assert evaluator.evaluate(expr) is None


class TestEvaluatorLogicalOperators:
    """Test evaluation of logical operations."""

    def test_and_both_true(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(
            left=Literal(True),
            operator='AND',
            right=Literal(True)
        )
        assert evaluator.evaluate(expr) is True

    def test_and_one_false(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(
            left=Literal(True),
            operator='AND',
            right=Literal(False)
        )
        assert evaluator.evaluate(expr) is False

    def test_and_short_circuit(self):
        node = Node(id=1, labels=['Person'], properties={'age': 20})
        evaluator = ExpressionEvaluator({'n': node})
        expr = BinaryOp(
            left=BinaryOp(PropertyAccess('n', 'age'), '<', Literal(25)),
            operator='AND',
            right=BinaryOp(PropertyAccess('n', 'name'), '=', Literal('Alice'))
        )
        # First condition is True, second accesses missing property
        # With tri-valued logic, True AND null yields null
        assert evaluator.evaluate(expr) is None

    def test_or_both_false(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(
            left=Literal(False),
            operator='OR',
            right=Literal(False)
        )
        assert evaluator.evaluate(expr) is False

    def test_or_one_true(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(
            left=Literal(True),
            operator='OR',
            right=Literal(False)
        )
        assert evaluator.evaluate(expr) is True

    def test_or_short_circuit(self):
        node = Node(id=1, labels=['Person'], properties={'age': 30})
        evaluator = ExpressionEvaluator({'n': node})
        expr = BinaryOp(
            left=BinaryOp(PropertyAccess('n', 'age'), '>', Literal(25)),
            operator='OR',
            right=BinaryOp(PropertyAccess('n', 'missing'), '=', Literal('x'))
        )
        # First condition is True, should short-circuit and not evaluate second
        assert evaluator.evaluate(expr) is True

    def test_not_true(self):
        evaluator = ExpressionEvaluator({})
        expr = UnaryOp(operator='NOT', operand=Literal(True))
        assert evaluator.evaluate(expr) is False

    def test_not_false(self):
        evaluator = ExpressionEvaluator({})
        expr = UnaryOp(operator='NOT', operand=Literal(False))
        assert evaluator.evaluate(expr) is True

    def test_not_null(self):
        evaluator = ExpressionEvaluator({})
        expr = UnaryOp(operator='NOT', operand=Literal(None))
        assert evaluator.evaluate(expr) is None

    def test_is_null(self):
        evaluator = ExpressionEvaluator({})
        expr = UnaryOp(operator='IS NULL', operand=Literal(None))
        assert evaluator.evaluate(expr) is True
        expr = UnaryOp(operator='IS NULL', operand=Literal(1))
        assert evaluator.evaluate(expr) is False

    def test_is_not_null(self):
        evaluator = ExpressionEvaluator({})
        expr = UnaryOp(operator='IS NOT NULL', operand=Literal(None))
        assert evaluator.evaluate(expr) is False
        expr = UnaryOp(operator='IS NOT NULL', operand=Literal(1))
        assert evaluator.evaluate(expr) is True


class TestEvaluatorComplexExpressions:
    """Test evaluation of complex nested expressions."""

    def test_complex_and_or(self):
        node = Node(id=1, labels=['Person'], properties={'age': 30, 'city': 'NYC'})
        evaluator = ExpressionEvaluator({'n': node})
        # n.age > 25 AND (n.city = 'NYC' OR n.city = 'LA')
        expr = BinaryOp(
            left=BinaryOp(PropertyAccess('n', 'age'), '>', Literal(25)),
            operator='AND',
            right=BinaryOp(
                left=BinaryOp(PropertyAccess('n', 'city'), '=', Literal('NYC')),
                operator='OR',
                right=BinaryOp(PropertyAccess('n', 'city'), '=', Literal('LA'))
            )
        )
        assert evaluator.evaluate(expr) is True

    def test_and_or_null_semantics(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(None), operator='AND', right=Literal(True))
        assert evaluator.evaluate(expr) is None
        expr = BinaryOp(left=Literal(None), operator='AND', right=Literal(False))
        assert evaluator.evaluate(expr) is False
        expr = BinaryOp(left=Literal(None), operator='OR', right=Literal(True))
        assert evaluator.evaluate(expr) is True
        expr = BinaryOp(left=Literal(None), operator='OR', right=Literal(False))
        assert evaluator.evaluate(expr) is None


class TestEvaluatorListExpressions:
    """Test list literals, IN, and list comprehensions."""

    def test_list_literal(self):
        evaluator = ExpressionEvaluator({})
        expr = ListLiteral(items=[Literal(1), Literal(2)])
        assert evaluator.evaluate(expr) == [1, 2]

    def test_in_operator_true(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(
            left=Literal(2),
            operator='IN',
            right=ListLiteral(items=[Literal(1), Literal(2)])
        )
        assert evaluator.evaluate(expr) is True

    def test_list_comprehension(self):
        evaluator = ExpressionEvaluator({})
        expr = ListComprehension(
            variable='x',
            list_expr=ListLiteral(items=[Literal(1), Literal(2), Literal(3)]),
            where_expr=BinaryOp(Variable('x'), '>', Literal(1)),
            projection=Variable('x')
        )
        assert evaluator.evaluate(expr) == [2, 3]

    def test_list_comprehension_nodes(self):
        alice = Node(id=1, labels=['Person'], properties={'name': 'Alice', 'age': 30})
        bob = Node(id=2, labels=['Person'], properties={'name': 'Bob', 'age': 20})
        evaluator = ExpressionEvaluator({'nodes': [alice, bob]})
        expr = ListComprehension(
            variable='n',
            list_expr=Variable('nodes'),
            where_expr=BinaryOp(PropertyAccess('n', 'age'), '>', Literal(25)),
            projection=PropertyAccess('n', 'name')
        )
        assert evaluator.evaluate(expr) == ['Alice']

    def test_list_comprehension_relationships(self):
        rel1 = Relationship(id=1, source_id=1, target_id=2, type='KNOWS', properties={'since': 2020})
        rel2 = Relationship(id=2, source_id=2, target_id=3, type='KNOWS', properties={'since': 2019})
        evaluator = ExpressionEvaluator({'rels': [rel1, rel2]})
        expr = ListComprehension(
            variable='r',
            list_expr=Variable('rels'),
            where_expr=BinaryOp(PropertyAccess('r', 'since'), '>=', Literal(2020)),
            projection=PropertyAccess('r', 'since')
        )
        assert evaluator.evaluate(expr) == [2020]

    def test_list_index(self):
        evaluator = ExpressionEvaluator({})
        expr = ListIndex(
            list_expr=ListLiteral(items=[Literal('a'), Literal('b')]),
            index_expr=Literal(1)
        )
        assert evaluator.evaluate(expr) == 'b'
        evaluator = ExpressionEvaluator({'m': {'a': 1}})
        expr = ListIndex(
            list_expr=Variable('m'),
            index_expr=Literal('a')
        )
        assert evaluator.evaluate(expr) == 1

    def test_list_concat(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(
            left=ListLiteral(items=[Literal(1)]),
            operator='+',
            right=ListLiteral(items=[Literal(2)])
        )
        assert evaluator.evaluate(expr) == [1, 2]

    def test_list_functions(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(name='size', arguments=[ListLiteral(items=[Literal(1), Literal(2)])])
        assert evaluator.evaluate(expr) == 2
        expr = FunctionCallExpression(name='head', arguments=[ListLiteral(items=[Literal(1), Literal(2)])])
        assert evaluator.evaluate(expr) == 1
        expr = FunctionCallExpression(name='tail', arguments=[ListLiteral(items=[Literal(1), Literal(2)])])
        assert evaluator.evaluate(expr) == [2]
        expr = FunctionCallExpression(name='last', arguments=[ListLiteral(items=[Literal(1), Literal(2)])])
        assert evaluator.evaluate(expr) == 2
        expr = FunctionCallExpression(name='reverse', arguments=[ListLiteral(items=[Literal(1), Literal(2)])])
        assert evaluator.evaluate(expr) == [2, 1]
        expr = FunctionCallExpression(name='range', arguments=[Literal(1), Literal(3)])
        assert evaluator.evaluate(expr) == [1, 2, 3]

    def test_list_slice(self):
        evaluator = ExpressionEvaluator({})
        expr = ListSlice(
            list_expr=ListLiteral(items=[Literal(1), Literal(2), Literal(3), Literal(4)]),
            start_expr=Literal(1),
            end_expr=Literal(3)
        )
        assert evaluator.evaluate(expr) == [2, 3]

    def test_list_predicates(self):
        evaluator = ExpressionEvaluator({})
        expr = ListPredicate(
            predicate='ANY',
            variable='x',
            list_expr=ListLiteral(items=[Literal(1), Literal(2)]),
            where_expr=BinaryOp(Variable('x'), '>', Literal(1))
        )
        assert evaluator.evaluate(expr) is True
        expr = ListPredicate(
            predicate='ALL',
            variable='x',
            list_expr=ListLiteral(items=[Literal(1), Literal(2)]),
            where_expr=BinaryOp(Variable('x'), '>', Literal(0))
        )
        assert evaluator.evaluate(expr) is True
        expr = ListPredicate(
            predicate='NONE',
            variable='x',
            list_expr=ListLiteral(items=[Literal(1), Literal(2)]),
            where_expr=BinaryOp(Variable('x'), '>', Literal(2))
        )
        assert evaluator.evaluate(expr) is True
        expr = ListPredicate(
            predicate='SINGLE',
            variable='x',
            list_expr=ListLiteral(items=[Literal(1), Literal(2)]),
            where_expr=BinaryOp(Variable('x'), '>', Literal(1))
        )
        assert evaluator.evaluate(expr) is True

    def test_list_predicates_without_where(self):
        evaluator = ExpressionEvaluator({})
        expr = ListPredicate(
            predicate='ANY',
            variable='x',
            list_expr=ListLiteral(items=[Literal(False), Literal(True)]),
            where_expr=None
        )
        assert evaluator.evaluate(expr) is True
        expr = ListPredicate(
            predicate='ALL',
            variable='x',
            list_expr=ListLiteral(items=[Literal(True), Literal(False)]),
            where_expr=None
        )
        assert evaluator.evaluate(expr) is False
        expr = ListPredicate(
            predicate='NONE',
            variable='x',
            list_expr=ListLiteral(items=[Literal(False), Literal(False)]),
            where_expr=None
        )
        assert evaluator.evaluate(expr) is True
        expr = ListPredicate(
            predicate='SINGLE',
            variable='x',
            list_expr=ListLiteral(items=[Literal(True), Literal(False), Literal(False)]),
            where_expr=None
        )
        assert evaluator.evaluate(expr) is True

    def test_list_predicates_with_nulls(self):
        evaluator = ExpressionEvaluator({})
        expr = ListPredicate(
            predicate='ANY',
            variable='x',
            list_expr=ListLiteral(items=[Literal(None), Literal(False)]),
            where_expr=None
        )
        assert evaluator.evaluate(expr) is None
        expr = ListPredicate(
            predicate='ALL',
            variable='x',
            list_expr=ListLiteral(items=[Literal(True), Literal(None)]),
            where_expr=None
        )
        assert evaluator.evaluate(expr) is None
        expr = ListPredicate(
            predicate='NONE',
            variable='x',
            list_expr=ListLiteral(items=[Literal(None), Literal(False)]),
            where_expr=None
        )
        assert evaluator.evaluate(expr) is None
        expr = ListPredicate(
            predicate='SINGLE',
            variable='x',
            list_expr=ListLiteral(items=[Literal(True), Literal(None)]),
            where_expr=None
        )
        assert evaluator.evaluate(expr) is None

    def test_list_predicates_with_null_where(self):
        evaluator = ExpressionEvaluator({})
        expr = ListPredicate(
            predicate='ANY',
            variable='x',
            list_expr=ListLiteral(items=[Literal(1)]),
            where_expr=Literal(None)
        )
        assert evaluator.evaluate(expr) is None

    def test_list_predicates_relationship_properties(self):
        rel1 = Relationship(id=1, source_id=1, target_id=2, type='KNOWS', properties={'since': 2019})
        rel2 = Relationship(id=2, source_id=2, target_id=3, type='KNOWS', properties={'since': 2021})
        evaluator = ExpressionEvaluator({'rels': [rel1, rel2]})
        expr = ListPredicate(
            predicate='ANY',
            variable='r',
            list_expr=Variable('rels'),
            where_expr=BinaryOp(PropertyAccess('r', 'since'), '>=', Literal(2021))
        )
        assert evaluator.evaluate(expr) is True
        expr = ListPredicate(
            predicate='ALL',
            variable='r',
            list_expr=Variable('rels'),
            where_expr=BinaryOp(PropertyAccess('r', 'since'), '>=', Literal(2019))
        )
        assert evaluator.evaluate(expr) is True
        expr = ListPredicate(
            predicate='NONE',
            variable='r',
            list_expr=Variable('rels'),
            where_expr=BinaryOp(PropertyAccess('r', 'since'), '<', Literal(2018))
        )
        assert evaluator.evaluate(expr) is True
        expr = ListPredicate(
            predicate='SINGLE',
            variable='r',
            list_expr=Variable('rels'),
            where_expr=BinaryOp(PropertyAccess('r', 'since'), '>=', Literal(2021))
        )
        assert evaluator.evaluate(expr) is True

    def test_list_predicates_mixed_direction_relationships(self):
        rel1 = Relationship(id=1, source_id=1, target_id=2, type='KNOWS', properties={'since': 2018})
        rel2 = Relationship(id=2, source_id=3, target_id=2, type='KNOWS', properties={'since': 2022})
        evaluator = ExpressionEvaluator({'rels': [rel1, rel2]})
        expr = ListPredicate(
            predicate='ALL',
            variable='r',
            list_expr=Variable('rels'),
            where_expr=BinaryOp(PropertyAccess('r', 'since'), '>=', Literal(2018))
        )
        assert evaluator.evaluate(expr) is True
        expr = ListPredicate(
            predicate='NONE',
            variable='r',
            list_expr=Variable('rels'),
            where_expr=BinaryOp(PropertyAccess('r', 'since'), '<', Literal(2018))
        )
        assert evaluator.evaluate(expr) is True
        expr = ListPredicate(
            predicate='SINGLE',
            variable='r',
            list_expr=Variable('rels'),
            where_expr=BinaryOp(PropertyAccess('r', 'since'), '>=', Literal(2022))
        )
        assert evaluator.evaluate(expr) is True

    def test_map_literal_and_access(self):
        evaluator = ExpressionEvaluator({})
        expr = MapLiteral(items={'a': Literal(1), 'b': Literal(2)})
        assert evaluator.evaluate(expr) == {'a': 1, 'b': 2}
        evaluator = ExpressionEvaluator({'m': {'a': 3}})
        expr = PropertyAccess(variable='m', property='a')
        assert evaluator.evaluate(expr) == 3

    def test_complex_with_not(self):
        node = Node(id=1, labels=['Person'], properties={'active': False})
        evaluator = ExpressionEvaluator({'n': node})
        # NOT n.active
        expr = UnaryOp(
            operator='NOT',
            operand=PropertyAccess('n', 'active')
        )
        assert evaluator.evaluate(expr) is True


class TestEvaluatorWhereFiltering:
    """Test WHERE-style expressions used after multi-hop matching."""

    def test_property_comparison_true(self):
        node = Node(id=1, labels=['Company'], properties={'size': 200})
        evaluator = ExpressionEvaluator({'c': node})
        expr = BinaryOp(PropertyAccess('c', 'size'), '>', Literal(100))
        assert evaluator.evaluate(expr) is True

    def test_property_comparison_false(self):
        node = Node(id=1, labels=['Company'], properties={'size': 50})
        evaluator = ExpressionEvaluator({'c': node})
        expr = BinaryOp(PropertyAccess('c', 'size'), '>', Literal(100))
        assert evaluator.evaluate(expr) is False


class TestEvaluatorCaseExpressions:
    """Test evaluation of CASE expressions."""

    def test_case_searched(self):
        node = Node(id=1, labels=['Person'], properties={'age': 20})
        evaluator = ExpressionEvaluator({'n': node})
        expr = CaseExpression(
            whens=[
                CaseWhen(
                    when_expr=BinaryOp(PropertyAccess('n', 'age'), '>', Literal(18)),
                    then_expr=Literal('adult')
                )
            ],
            else_expr=Literal('minor')
        )
        assert evaluator.evaluate(expr) == 'adult'


class TestEvaluatorTemporalFunctions:
    """Test evaluation of temporal functions and operations."""

    def test_temporal_parsing(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(name='date', arguments=[Literal('2024-01-02')])
        assert evaluator.evaluate(expr).isoformat() == '2024-01-02'
        expr = FunctionCallExpression(name='time', arguments=[Literal('10:11:12')])
        assert evaluator.evaluate(expr).isoformat() == '10:11:12'
        expr = FunctionCallExpression(name='datetime', arguments=[Literal('2024-01-02T10:11:12')])
        assert evaluator.evaluate(expr).isoformat() == '2024-01-02T10:11:12'
        expr = FunctionCallExpression(name='duration', arguments=[Literal('P1DT2H')])
        assert evaluator.evaluate(expr).total_seconds() == 93600

    def test_temporal_arithmetic(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(
            left=FunctionCallExpression(name='date', arguments=[Literal('2024-01-01')]),
            operator='+',
            right=FunctionCallExpression(name='duration', arguments=[Literal('P2D')])
        )
        assert evaluator.evaluate(expr).isoformat() == '2024-01-03'
        expr = BinaryOp(
            left=FunctionCallExpression(name='datetime', arguments=[Literal('2024-01-01T00:00:00')]),
            operator='-',
            right=FunctionCallExpression(name='duration', arguments=[Literal('PT1H')])
        )
        assert evaluator.evaluate(expr).isoformat() == '2023-12-31T23:00:00'

    def test_temporal_truncate_and_between(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(
            name='date.truncate',
            arguments=[Literal('month'), FunctionCallExpression(name='date', arguments=[Literal('2024-02-20')])]
        )
        assert evaluator.evaluate(expr).isoformat() == '2024-02-01'
        expr = FunctionCallExpression(
            name='duration.between',
            arguments=[
                FunctionCallExpression(name='date', arguments=[Literal('2024-01-01')]),
                FunctionCallExpression(name='date', arguments=[Literal('2024-01-03')])
            ]
        )
        assert evaluator.evaluate(expr).total_seconds() == 172800


class TestEvaluatorTriValuedLogic:
    """Test tri-valued (null) logic."""

    def test_in_with_nulls(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(1), operator='IN', right=ListLiteral(items=[Literal(1), Literal(None)]))
        assert evaluator.evaluate(expr) is True
        expr = BinaryOp(left=Literal(2), operator='IN', right=ListLiteral(items=[Literal(1), Literal(None)]))
        assert evaluator.evaluate(expr) is None
        expr = BinaryOp(left=Literal(None), operator='IN', right=ListLiteral(items=[Literal(1), Literal(2)]))
        assert evaluator.evaluate(expr) is None
        expr = BinaryOp(left=Literal(None), operator='IN', right=ListLiteral(items=[Literal(None)]))
        assert evaluator.evaluate(expr) is None

    def test_null_comparisons(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(None), operator='=', right=Literal(None))
        assert evaluator.evaluate(expr) is None
        expr = BinaryOp(left=Literal(None), operator='!=', right=Literal(None))
        assert evaluator.evaluate(expr) is None
        expr = BinaryOp(left=Literal(None), operator='<', right=Literal(1))
        assert evaluator.evaluate(expr) is None

    def test_and_or_nulls(self):
        evaluator = ExpressionEvaluator({})
        expr = BinaryOp(left=Literal(True), operator='AND', right=Literal(None))
        assert evaluator.evaluate(expr) is None
        expr = BinaryOp(left=Literal(False), operator='AND', right=Literal(None))
        assert evaluator.evaluate(expr) is False
        expr = BinaryOp(left=Literal(True), operator='OR', right=Literal(None))
        assert evaluator.evaluate(expr) is True
        expr = BinaryOp(left=Literal(False), operator='OR', right=Literal(None))
        assert evaluator.evaluate(expr) is None


class TestEvaluatorCasting:
    """Test explicit casting functions."""

    def test_casting_functions(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(name='toInteger', arguments=[Literal('42')])
        assert evaluator.evaluate(expr) == 42
        expr = FunctionCallExpression(name='toInteger', arguments=[Literal('3.9')])
        assert evaluator.evaluate(expr) == 3
        expr = FunctionCallExpression(name='toInteger', arguments=[Literal(True)])
        assert evaluator.evaluate(expr) == 1
        expr = FunctionCallExpression(name='toFloat', arguments=[Literal('3.5')])
        assert evaluator.evaluate(expr) == 3.5
        expr = FunctionCallExpression(name='toFloat', arguments=[Literal(2)])
        assert evaluator.evaluate(expr) == 2.0
        expr = FunctionCallExpression(name='toString', arguments=[Literal(123)])
        assert evaluator.evaluate(expr) == '123'
        expr = FunctionCallExpression(name='toInteger', arguments=[Literal(None)])
        assert evaluator.evaluate(expr) is None


class TestEvaluatorStringFunctions:
    """Test string function behavior."""

    def test_string_functions(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(name='toUpper', arguments=[Literal('hi')])
        assert evaluator.evaluate(expr) == 'HI'
        expr = FunctionCallExpression(name='toLower', arguments=[Literal('HI')])
        assert evaluator.evaluate(expr) == 'hi'
        expr = FunctionCallExpression(name='trim', arguments=[Literal('  hi  ')])
        assert evaluator.evaluate(expr) == 'hi'
        expr = FunctionCallExpression(name='split', arguments=[Literal('a,b,c'), Literal(',')])
        assert evaluator.evaluate(expr) == ['a', 'b', 'c']
        expr = FunctionCallExpression(name='substring', arguments=[Literal('abcdef'), Literal(1), Literal(3)])
        assert evaluator.evaluate(expr) == 'bcd'
        expr = FunctionCallExpression(name='substring', arguments=[Literal('abcdef'), Literal(2)])
        assert evaluator.evaluate(expr) == 'cdef'
        expr = FunctionCallExpression(name='regex', arguments=[Literal('abc'), Literal('a.c')])
        assert evaluator.evaluate(expr) is True
        expr = FunctionCallExpression(name='matches', arguments=[Literal('abc'), Literal('^d')])
        assert evaluator.evaluate(expr) is False

    def test_string_functions_with_null(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(name='toUpper', arguments=[Literal(None)])
        assert evaluator.evaluate(expr) is None
        expr = FunctionCallExpression(name='split', arguments=[Literal(None), Literal(',')])
        assert evaluator.evaluate(expr) is None
        expr = FunctionCallExpression(name='substring', arguments=[Literal(None), Literal(1)])
        assert evaluator.evaluate(expr) is None
        expr = FunctionCallExpression(name='matches', arguments=[Literal(None), Literal('a')])
        assert evaluator.evaluate(expr) is None

    def test_string_functions_invalid_types(self):
        evaluator = ExpressionEvaluator({})
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='split', arguments=[Literal(1), Literal(',')]))
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='substring', arguments=[Literal('a'), Literal('1')]))
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='regex', arguments=[Literal('a'), Literal(1)]))

    def test_string_functions_bounds(self):
        evaluator = ExpressionEvaluator({})
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='substring', arguments=[Literal('abc'), Literal(-1)]))
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(
                name='substring',
                arguments=[Literal('abc'), Literal(1), Literal(-2)]
            ))

    def test_string_functions_invalid_regex(self):
        evaluator = ExpressionEvaluator({})
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='regex', arguments=[Literal('abc'), Literal('[')]))


class TestEvaluatorTextFunctions:
    """Test extended text functions."""

    def test_text_cleanup_functions(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(name='deaccent', arguments=[Literal('caf\u00e9')])
        assert evaluator.evaluate(expr) == 'cafe'
        expr = FunctionCallExpression(name='strip_html', arguments=[Literal('<b>Hi</b> <i>there</i>')])
        assert evaluator.evaluate(expr) == 'Hi there'
        expr = FunctionCallExpression(name='strip_emoji', arguments=[Literal('Hi \U0001F600')])
        assert evaluator.evaluate(expr) == 'Hi '

    def test_snake_case_function(self):
        evaluator = ExpressionEvaluator({})
        # Basic spaces
        expr = FunctionCallExpression(name='snake_case', arguments=[Literal('Hello World')])
        assert evaluator.evaluate(expr) == 'hello_world'
        # CamelCase
        expr = FunctionCallExpression(name='snake_case', arguments=[Literal('camelCaseText')])
        assert evaluator.evaluate(expr) == 'camel_case_text'
        # PascalCase
        expr = FunctionCallExpression(name='snake_case', arguments=[Literal('PascalCaseText')])
        assert evaluator.evaluate(expr) == 'pascal_case_text'
        # With accents (should be removed for ASCII compatibility)
        expr = FunctionCallExpression(name='snake_case', arguments=[Literal('Caf\u00e9 Cr\u00e8me')])
        assert evaluator.evaluate(expr) == 'cafe_creme'
        # Special characters
        expr = FunctionCallExpression(name='snake_case', arguments=[Literal('user@email.com')])
        assert evaluator.evaluate(expr) == 'user_email_com'
        # Multiple spaces/special chars collapse to single underscore
        expr = FunctionCallExpression(name='snake_case', arguments=[Literal('hello   world!!!')])
        assert evaluator.evaluate(expr) == 'hello_world'
        # Leading/trailing special chars are stripped
        expr = FunctionCallExpression(name='snake_case', arguments=[Literal('  Hello World  ')])
        assert evaluator.evaluate(expr) == 'hello_world'
        # Null handling
        expr = FunctionCallExpression(name='snake_case', arguments=[Literal(None)])
        assert evaluator.evaluate(expr) is None

    def test_text_similarity_functions(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(
            name='levenshtein',
            arguments=[Literal('kitten'), Literal('sitting')]
        )
        assert evaluator.evaluate(expr) == 3
        expr = FunctionCallExpression(
            name='jaccard',
            arguments=[Literal('a b c'), Literal('b c d')]
        )
        assert evaluator.evaluate(expr) == 0.5

    def test_text_functions_with_null(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(name='deaccent', arguments=[Literal(None)])
        assert evaluator.evaluate(expr) is None
        expr = FunctionCallExpression(
            name='levenshtein',
            arguments=[Literal(None), Literal('a')]
        )
        assert evaluator.evaluate(expr) is None

    def test_text_functions_invalid_types(self):
        evaluator = ExpressionEvaluator({})
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='deaccent', arguments=[Literal(1)]))
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(
                name='jaccard',
                arguments=[Literal('a'), Literal(1)]
            ))

class TestEvaluatorCoalesce:
    """Test coalesce()."""

    def test_coalesce(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(
            name='coalesce',
            arguments=[Literal(None), Literal('a'), Literal('b')]
        )
        assert evaluator.evaluate(expr) == 'a'
        expr = FunctionCallExpression(
            name='coalesce',
            arguments=[Literal(None), Literal(None)]
        )
        assert evaluator.evaluate(expr) is None


class TestEvaluatorMapListFunctions:
    """Test keys/values and apoc-style helpers."""

    def test_keys_values_for_map(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(name='keys', arguments=[MapLiteral(items={'a': Literal(1), 'b': Literal(2)})])
        assert evaluator.evaluate(expr) == ['a', 'b']
        expr = FunctionCallExpression(name='values', arguments=[MapLiteral(items={'a': Literal(1), 'b': Literal(2)})])
        assert evaluator.evaluate(expr) == [1, 2]

    def test_keys_values_for_node_relationship(self):
        node = Node(id=1, labels=['Person'], properties={'name': 'Alice', 'age': 30})
        rel = Relationship(id=1, source_id=1, target_id=2, type='KNOWS', properties={'since': 2020})
        evaluator = ExpressionEvaluator({'n': node, 'r': rel})
        expr = FunctionCallExpression(name='keys', arguments=[Variable('n')])
        assert sorted(evaluator.evaluate(expr)) == ['age', 'name']
        expr = FunctionCallExpression(name='values', arguments=[Variable('r')])
        assert evaluator.evaluate(expr) == [2020]

    def test_apoc_text_join_split(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(
            name='apoc.text.join',
            arguments=[ListLiteral(items=[Literal('a'), Literal('b')]), Literal(',')]
        )
        assert evaluator.evaluate(expr) == 'a,b'
        expr = FunctionCallExpression(
            name='apoc.text.split',
            arguments=[Literal('a,b,c'), Literal(',')]
        )
        assert evaluator.evaluate(expr) == ['a', 'b', 'c']
        expr = FunctionCallExpression(
            name='apoc.text.replace',
            arguments=[Literal('a-b-c'), Literal('-'), Literal('_')]
        )
        assert evaluator.evaluate(expr) == 'a_b_c'
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='apoc.text.split', arguments=[Literal('a'), Literal('[')]))

    def test_apoc_map_helpers(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(
            name='apoc.map.merge',
            arguments=[MapLiteral(items={'a': Literal(1)}), MapLiteral(items={'b': Literal(2)})]
        )
        assert evaluator.evaluate(expr) == {'a': 1, 'b': 2}
        expr = FunctionCallExpression(
            name='apoc.convert.toMap',
            arguments=[MapLiteral(items={'a': Literal(1)})]
        )
        assert evaluator.evaluate(expr) == {'a': 1}
        expr = FunctionCallExpression(
            name='apoc.map.fromPairs',
            arguments=[ListLiteral(items=[ListLiteral(items=[Literal('a'), Literal(1)])])]
        )
        assert evaluator.evaluate(expr) == {'a': 1}
        expr = FunctionCallExpression(
            name='apoc.map.clean',
            arguments=[
                MapLiteral(items={'a': Literal(1), 'b': Literal(None), 'c': Literal('')}),
                ListLiteral(items=[]),
                ListLiteral(items=[Literal(None), Literal('')])
            ]
        )
        assert evaluator.evaluate(expr) == {'a': 1}
        expr = FunctionCallExpression(
            name='apoc.map.removeKey',
            arguments=[MapLiteral(items={'a': Literal(1), 'b': Literal(2)}), Literal('a')]
        )
        assert evaluator.evaluate(expr) == {'b': 2}
        expr = FunctionCallExpression(
            name='apoc.map.get',
            arguments=[MapLiteral(items={'a': Literal(1)}), Literal('a'), Literal(9)]
        )
        assert evaluator.evaluate(expr) == 1
        expr = FunctionCallExpression(
            name='apoc.map.get',
            arguments=[MapLiteral(items={'a': Literal(1)}), Literal('missing'), Literal(9)]
        )
        assert evaluator.evaluate(expr) == 9

    def test_apoc_coll_helpers(self):
        evaluator = ExpressionEvaluator({})
        expr = FunctionCallExpression(
            name='apoc.coll.contains',
            arguments=[ListLiteral(items=[Literal(1), Literal(2)]), Literal(2)]
        )
        assert evaluator.evaluate(expr) is True
        expr = FunctionCallExpression(
            name='apoc.coll.toSet',
            arguments=[ListLiteral(items=[Literal(1), Literal(1), Literal(2)])]
        )
        assert evaluator.evaluate(expr) == [1, 2]

    def test_apoc_helpers_invalid_types(self):
        evaluator = ExpressionEvaluator({})
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='keys', arguments=[Literal(1)]))
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='apoc.text.join', arguments=[Literal('a'), Literal(',')]))
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='apoc.map.merge', arguments=[Literal(1), Literal(2)]))
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='apoc.map.clean', arguments=[Literal(1), Literal([]), Literal([])]))
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='apoc.map.fromPairs', arguments=[Literal(1)]))
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='apoc.convert.toMap', arguments=[Literal(1)]))
        with pytest.raises(CypherExecutionError):
            evaluator.evaluate(FunctionCallExpression(name='apoc.coll.contains', arguments=[Literal(1), Literal(2)]))

class TestEvaluatorGraphFunctions:
    """Test labels/type/properties/id/elementId over nodes and relationships."""

    def _env(self):
        node = Node(id=7, labels=['Person', 'Admin'], properties={'name': 'Alice', 'age': 30})
        rel = Relationship(id=3, source_id=7, target_id=8, type='KNOWS', properties={'since': 2020})
        return node, rel, ExpressionEvaluator({'n': node, 'r': rel})

    def test_labels(self):
        node, _, ev = self._env()
        assert ev.evaluate(FunctionCallExpression(name='labels', arguments=[Variable('n')])) == ['Person', 'Admin']

    def test_type(self):
        _, _, ev = self._env()
        assert ev.evaluate(FunctionCallExpression(name='type', arguments=[Variable('r')])) == 'KNOWS'

    def test_properties_node_and_relationship(self):
        _, _, ev = self._env()
        assert ev.evaluate(FunctionCallExpression(name='properties', arguments=[Variable('n')])) == {'name': 'Alice', 'age': 30}
        assert ev.evaluate(FunctionCallExpression(name='properties', arguments=[Variable('r')])) == {'since': 2020}

    def test_properties_of_map_returns_map(self):
        ev = ExpressionEvaluator({})
        expr = FunctionCallExpression(name='properties', arguments=[MapLiteral(items={'a': Literal(1)})])
        assert ev.evaluate(expr) == {'a': 1}

    def test_id_and_element_id(self):
        _, _, ev = self._env()
        assert ev.evaluate(FunctionCallExpression(name='id', arguments=[Variable('n')])) == 7
        assert ev.evaluate(FunctionCallExpression(name='id', arguments=[Variable('r')])) == 3
        # elementId() returns the identifier as a string.
        assert ev.evaluate(FunctionCallExpression(name='elementId', arguments=[Variable('n')])) == '7'

    def test_null_propagation(self):
        ev = ExpressionEvaluator({})
        for fn in ('labels', 'type', 'properties', 'id', 'elementId'):
            assert ev.evaluate(FunctionCallExpression(name=fn, arguments=[Literal(None)])) is None

    def test_wrong_type_raises(self):
        _, _, ev = self._env()
        # labels() on a relationship, type() on a node, id() on a scalar.
        with pytest.raises(CypherExecutionError):
            ev.evaluate(FunctionCallExpression(name='labels', arguments=[Variable('r')]))
        with pytest.raises(CypherExecutionError):
            ev.evaluate(FunctionCallExpression(name='type', arguments=[Variable('n')]))
        with pytest.raises(CypherExecutionError):
            ev.evaluate(FunctionCallExpression(name='id', arguments=[Literal(1)]))


class TestEvaluatorListFunctions:
    """Test filter/extract/reduce list functions."""

    def test_filter_extract_reduce(self):
        evaluator = ExpressionEvaluator({'xs': [1, 2, 3]})
        expr = ListComprehension(
            variable='x',
            list_expr=Variable('xs'),
            where_expr=BinaryOp(Variable('x'), '>', Literal(1)),
            projection=Variable('x')
        )
        assert evaluator.evaluate(expr) == [2, 3]
        expr = ListComprehension(
            variable='x',
            list_expr=Variable('xs'),
            where_expr=None,
            projection=BinaryOp(Variable('x'), '+', Literal(1))
        )
        assert evaluator.evaluate(expr) == [2, 3, 4]
        expr = ReduceExpression(
            accumulator='acc',
            init_expr=Literal(0),
            variable='x',
            list_expr=Variable('xs'),
            expression=BinaryOp(Variable('acc'), '+', Variable('x'))
        )
        assert evaluator.evaluate(expr) == 6

    def test_path_functions(self):
        node_a = Node(id=1, labels=['Person'], properties={'name': 'A'})
        node_b = Node(id=2, labels=['Person'], properties={'name': 'B'})
        rel = Relationship(id=1, source_id=1, target_id=2, type='KNOWS', properties={})
        path = Path(nodes=[node_a, node_b], relationships=[rel])
        evaluator = ExpressionEvaluator({'p': path})
        expr = FunctionCallExpression(name='nodes', arguments=[Variable('p')])
        assert evaluator.evaluate(expr) == [node_a, node_b]
        expr = FunctionCallExpression(name='relationships', arguments=[Variable('p')])
        assert evaluator.evaluate(expr) == [rel]

    def test_point_and_distance(self):
        evaluator = ExpressionEvaluator({})
        point_expr = FunctionCallExpression(
            name='point',
            arguments=[MapLiteral(items={'x': Literal(0), 'y': Literal(3)})]
        )
        point_expr2 = FunctionCallExpression(
            name='point',
            arguments=[MapLiteral(items={'x': Literal(0), 'y': Literal(0)})]
        )
        p1 = evaluator.evaluate(point_expr)
        p2 = evaluator.evaluate(point_expr2)
        dist_expr = FunctionCallExpression(name='distance', arguments=[Literal(p1), Literal(p2)])
        assert evaluator.evaluate(dist_expr) == 3.0

        wgs_expr = FunctionCallExpression(
            name='point',
            arguments=[MapLiteral(items={'longitude': Literal(0), 'latitude': Literal(0)})]
        )
        wgs_expr2 = FunctionCallExpression(
            name='point',
            arguments=[MapLiteral(items={'longitude': Literal(0), 'latitude': Literal(1)})]
        )
        w1 = evaluator.evaluate(wgs_expr)
        w2 = evaluator.evaluate(wgs_expr2)
        dist_expr = FunctionCallExpression(name='distance', arguments=[Literal(w1), Literal(w2)])
        assert evaluator.evaluate(dist_expr) == pytest.approx(111195, rel=1e-3)

    def test_case_simple(self):
        node = Node(id=1, labels=['Person'], properties={'role': 'admin'})
        evaluator = ExpressionEvaluator({'n': node})
        expr = CaseExpression(
            base_expr=PropertyAccess('n', 'role'),
            whens=[
                CaseWhen(when_expr=Literal('admin'), then_expr=Literal(1)),
                CaseWhen(when_expr=Literal('user'), then_expr=Literal(0)),
            ],
            else_expr=Literal(-1)
        )
        assert evaluator.evaluate(expr) == 1

    def test_case_simple_null_does_not_match(self):
        node = Node(id=1, labels=['Person'], properties={})
        evaluator = ExpressionEvaluator({'n': node})
        expr = CaseExpression(
            base_expr=PropertyAccess('n', 'missing'),
            whens=[CaseWhen(when_expr=Literal(None), then_expr=Literal(1))],
            else_expr=Literal(0)
        )
        assert evaluator.evaluate(expr) == 0

    def test_case_multiple_when(self):
        node = Node(id=1, labels=['Person'], properties={'age': 16})
        evaluator = ExpressionEvaluator({'n': node})
        expr = CaseExpression(
            whens=[
                CaseWhen(
                    when_expr=BinaryOp(PropertyAccess('n', 'age'), '<', Literal(13)),
                    then_expr=Literal('child')
                ),
                CaseWhen(
                    when_expr=BinaryOp(PropertyAccess('n', 'age'), '<', Literal(20)),
                    then_expr=Literal('teen')
                ),
            ],
            else_expr=Literal('adult')
        )
        assert evaluator.evaluate(expr) == 'teen'

    def test_case_no_else_returns_none(self):
        node = Node(id=1, labels=['Person'], properties={'active': False})
        evaluator = ExpressionEvaluator({'n': node})
        expr = CaseExpression(
            whens=[
                CaseWhen(when_expr=PropertyAccess('n', 'active'), then_expr=Literal('yes'))
            ]
        )
        assert evaluator.evaluate(expr) is None


class TestEvaluatorPatternComprehension:
    """Test pattern comprehension evaluation hooks."""

    def test_pattern_comprehension_requires_matcher(self):
        pattern = Pattern(elements=[PatternElement(node=NodePattern(variable='a'))])
        expr = PatternComprehension(pattern=pattern, projection=Variable('a'))
        evaluator = ExpressionEvaluator({})
        with pytest.raises(CypherExecutionError) as exc_info:
            evaluator.evaluate(expr)
        assert "Pattern comprehensions require a query context" in str(exc_info.value)

    def test_pattern_comprehension_delegates_to_matcher(self):
        pattern = Pattern(elements=[
            PatternElement(
                node=NodePattern(variable='a'),
                relationship=RelationshipPattern(rel_type='KNOWS')
            ),
            PatternElement(node=NodePattern(variable='b'))
        ])
        expr = PatternComprehension(pattern=pattern, projection=Variable('b'))

        def matcher(pattern_expr, context):
            assert pattern_expr.pattern == pattern
            assert context['seed'] == 5
            return [context['seed'], 7]

        evaluator = ExpressionEvaluator({'seed': 5}, pattern_matcher=matcher)
        assert evaluator.evaluate(expr) == [5, 7]
