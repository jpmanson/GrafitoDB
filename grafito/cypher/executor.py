"""Executor for Cypher queries - translates AST to Grafito API calls."""

from __future__ import annotations

import base64
import csv
import hashlib
from html.parser import HTMLParser
import io
import orjson
import os
import re
import urllib.parse
import urllib.request
from typing import Any
from ..database import GrafitoDatabase
from ..models import Node, Relationship, Path
from .ast_nodes import (
    Query, CreateClause, MergeClause, MatchClause, WithClause, WhereClause, ReturnClause, SetClause, RemoveClause,
    Pattern, PatternElement, NodePattern, RelationshipPattern, PatternFunction,
    ReturnItem, PropertyAccess, PropertyLookup, Literal, FunctionCall, SubqueryClause, ProcedureCallClause, Variable, UnwindClause,
    LoadCsvClause,
    Expression, BinaryOp,
    PatternComprehension, CreateIndexClause, DropIndexClause, ShowIndexesClause,
    CreateConstraintClause, DropConstraintClause, ShowConstraintsClause,
    ForeachClause
)
from .evaluator import ExpressionEvaluator
from .exceptions import CypherExecutionError
from ..filters import PropertyFilter, PropertyFilterGroup


class _HtmlNode:
    """Simple HTML node for selector traversal."""

    def __init__(self, tag: str, attrs: list[tuple[str, str]] | dict[str, str], parent: '_HtmlNode' | None = None):
        self.tag = tag.lower()
        if isinstance(attrs, dict):
            self.attrs = {key.lower(): value for key, value in attrs.items()}
        else:
            self.attrs = {key.lower(): value for key, value in attrs}
        self.parent = parent
        self.children: list[_HtmlNode] = []
        self.text_parts: list[str] = []

    def text_content(self) -> str:
        parts = list(self.text_parts)
        for child in self.children:
            parts.append(child.text_content())
        return "".join(parts)

    def classes(self) -> list[str]:
        class_value = self.attrs.get("class") or ""
        return class_value.split()


class _HtmlTreeBuilder(HTMLParser):
    """Minimal HTML parser that builds a tree for selectors."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _HtmlNode("document", {})
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        node = _HtmlNode(tag, attrs, parent=self.stack[-1])
        self.stack[-1].children.append(node)
        self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        node = _HtmlNode(tag, attrs, parent=self.stack[-1])
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for idx in range(len(self.stack) - 1, 0, -1):
            if self.stack[idx].tag == tag:
                del self.stack[idx:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].text_parts.append(data)

    def error(self, message: str) -> None:
        return


class CypherExecutor:
    """Executes Cypher query AST against a GrafitoDatabase."""

    def __init__(self, db: GrafitoDatabase):
        self.db = db

    def _make_evaluator(self, context: dict[str, Any]) -> ExpressionEvaluator:
        """Build an expression evaluator with pattern comprehension support."""
        return ExpressionEvaluator(context, pattern_matcher=self._pattern_comprehension_matcher)

    def _evaluate_properties(
        self,
        properties: dict[str, Any] | None,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Evaluate property map expressions against the current context."""
        if not properties:
            return {}
        evaluator = self._make_evaluator(context or {})
        evaluated: dict[str, Any] = {}
        for key, expr in properties.items():
            if isinstance(expr, Expression):
                evaluated[key] = evaluator.evaluate(expr)
            else:
                evaluated[key] = expr
        return evaluated

    def _pattern_comprehension_matcher(self, expr: PatternComprehension, context: dict[str, Any]) -> list[Any]:
        """Evaluate a pattern comprehension against the current context."""
        return self._evaluate_pattern_comprehension(expr, context)

    def _evaluate_pattern_comprehension(self, expr: PatternComprehension, context: dict[str, Any]) -> list[Any]:
        """Evaluate a pattern comprehension by matching the pattern and projecting results."""
        matches = self._match_pattern(expr.pattern)
        results = []

        for match in matches:
            if not self._pattern_bindings_match(context, match):
                continue

            merged = context.copy()
            merged.update(match)
            evaluator = self._make_evaluator(merged)

            if expr.where_expr is not None:
                try:
                    if not evaluator.evaluate(expr.where_expr):
                        continue
                except CypherExecutionError:
                    continue

            results.append(self._serialize_value(evaluator.evaluate(expr.projection)))

        return results

    def _pattern_bindings_match(self, context: dict[str, Any], match: dict[str, Any]) -> bool:
        """Check if matched variables are compatible with existing context bindings."""
        for name, value in match.items():
            if name not in context:
                continue
            if not self._same_entity(context[name], value):
                return False
        return True

    def _same_entity(self, left: Any, right: Any) -> bool:
        """Compare two bound entities by id when possible."""
        if left is right:
            return True
        left_id = self._entity_id(left)
        right_id = self._entity_id(right)
        if left_id is not None and right_id is not None:
            return left_id == right_id
        return left == right

    def _entity_id(self, value: Any) -> Any:
        """Extract an entity id from nodes/relationships or serialized dicts."""
        if hasattr(value, 'id'):
            return value.id
        if isinstance(value, dict) and 'id' in value:
            return value['id']
        return None

    def execute(self, query: Query) -> list[dict]:
        """Execute a query and return results.

        Args:
            query: Parsed Query AST

        Returns:
            List of result dictionaries

        Raises:
            CypherExecutionError: If execution fails
        """
        if query.union_clauses:
            return self._execute_union(query)

        if isinstance(query.clause, SubqueryClause):
            return self._execute_subquery(query.clause, [])
        if isinstance(query.clause, ProcedureCallClause):
            return self._execute_procedure_call(query.clause, [])

        # Check if multi-clause query (with WITH)
        if query.clauses:
            return self._execute_multi_clause(query.clauses, initial_results=None)

        # Single clause query
        if isinstance(query.clause, CreateClause):
            return self._execute_create(query.clause)
        elif isinstance(query.clause, MergeClause):
            return self._execute_merge(query.clause)
        elif isinstance(query.clause, MatchClause):
            return self._execute_match(query.clause)
        elif isinstance(query.clause, UnwindClause):
            return self._execute_unwind(query.clause, [{}])
        elif isinstance(query.clause, WithClause):
            return self._execute_with(query.clause, [{}])
        elif isinstance(query.clause, CreateIndexClause):
            return self._execute_create_index(query.clause)
        elif isinstance(query.clause, DropIndexClause):
            return self._execute_drop_index(query.clause)
        elif isinstance(query.clause, ShowIndexesClause):
            return self._execute_show_indexes(query.clause)
        elif isinstance(query.clause, CreateConstraintClause):
            return self._execute_create_constraint(query.clause)
        elif isinstance(query.clause, DropConstraintClause):
            return self._execute_drop_constraint(query.clause)
        elif isinstance(query.clause, ShowConstraintsClause):
            return self._execute_show_constraints(query.clause)
        elif isinstance(query.clause, ForeachClause):
            return self._execute_foreach(query.clause, [{}])
        elif isinstance(query.clause, ReturnClause):
            # Standalone RETURN (no preceding MATCH/WITH): project against a
            # single empty row, e.g. `RETURN 1 + 1 AS x`.
            return self._apply_return([{}], query.clause)
        else:
            raise CypherExecutionError(f"Unknown clause type: {type(query.clause)}")

    def _execute_union(self, query: Query) -> list[dict]:
        """Execute UNION/UNION ALL queries."""
        results = self.execute(Query(clause=query.clause, clauses=query.clauses))

        for union_clause in query.union_clauses:
            union_results = self.execute(union_clause.query)
            if union_clause.all:
                results.extend(union_results)
            else:
                results = self._union_distinct(results, union_results)

        return results

    def _union_distinct(self, left: list[dict], right: list[dict]) -> list[dict]:
        """Return union of two result sets with distinct rows."""
        seen = {self._freeze_result(row) for row in left}
        combined = left[:]
        for row in right:
            frozen = self._freeze_result(row)
            if frozen in seen:
                continue
            seen.add(frozen)
            combined.append(row)
        return combined

    def _freeze_result(self, value: Any) -> Any:
        """Convert results to hashable structures for UNION distinct."""
        if isinstance(value, dict):
            return tuple(sorted((k, self._freeze_result(v)) for k, v in value.items()))
        if isinstance(value, list):
            return tuple(self._freeze_result(v) for v in value)
        return value

    def _execute_multi_clause(self, clauses: list, initial_results: list[dict] | None = None) -> list[dict]:
        """Execute multi-clause query with WITH pipeline.

        Args:
            clauses: List of clause AST nodes (MATCH, WITH, etc.)

        Returns:
            Final result set

        The WITH clause acts as a pipeline:
        - MATCH (n) returns results
        - WITH filters/transforms those results
        - Next MATCH uses WITH results as context
        """
        results = initial_results if initial_results is not None else []

        for i, clause in enumerate(clauses):
            if not results and isinstance(
                clause,
                (WithClause, UnwindClause, LoadCsvClause, ProcedureCallClause, ForeachClause, SubqueryClause),
            ):
                results = [{}]
            if isinstance(clause, MatchClause):
                # Execute MATCH with current context
                new_results = self._execute_match(clause, context=results)
                results = new_results
            elif isinstance(clause, CreateClause):
                new_results = self._execute_create(clause, context=results if results else None)
                results = new_results
            elif isinstance(clause, MergeClause):
                # Execute MERGE
                new_results = self._execute_merge(clause, context=results if results else None)
                results = new_results
            elif isinstance(clause, WithClause):
                # Apply WITH transformation to current results
                results = self._execute_with(clause, results)
            elif isinstance(clause, UnwindClause):
                results = self._execute_unwind(clause, results)
            elif isinstance(clause, LoadCsvClause):
                results = self._execute_load_csv(clause, results)
            elif isinstance(clause, ForeachClause):
                results = self._execute_foreach(clause, results)
            elif isinstance(clause, SetClause):
                self._execute_set(results, clause)
            elif isinstance(clause, ReturnClause):
                results = self._apply_return(results, clause)
            elif isinstance(clause, SubqueryClause):
                results = self._execute_subquery(clause, results)
            elif isinstance(clause, ProcedureCallClause):
                results = self._execute_procedure_call(clause, results)
            else:
                raise CypherExecutionError(f"Unknown clause type in multi-clause query: {type(clause)}")

        return results

    def _execute_subquery(self, clause: SubqueryClause, input_results: list[dict]) -> list[dict]:
        """Execute CALL { ... } subquery with scoped variables."""
        if not input_results:
            input_results = [{}]

        output = []
        for row in input_results:
            sub_results = self._execute_query_with_context(clause.query, [row])
            for sub_row in sub_results:
                merged = row.copy()
                merged.update(sub_row)
                output.append(merged)

        return output

    def _execute_procedure_call(
        self,
        clause: ProcedureCallClause,
        input_results: list[dict],
    ) -> list[dict]:
        """Execute CALL procedure with optional YIELD projection."""
        if not input_results:
            input_results = [{}]

        output = []
        for row in input_results:
            evaluator = self._make_evaluator(row)
            args = [evaluator.evaluate(arg) for arg in clause.arguments]
            proc_results = self._invoke_procedure(clause.name, args)
            if not proc_results:
                continue
            for proc_row in proc_results:
                if clause.yield_items:
                    projected = {key: proc_row.get(key) for key in clause.yield_items}
                else:
                    projected = proc_row
                merged = row.copy()
                merged.update(projected)
                output.append(merged)
        return output

    def _invoke_procedure(self, name: str, args: list[Any]) -> list[dict]:
        """Dispatch supported procedures by name."""
        lower_name = name.lower()
        if lower_name == "db.vector.search":
            if len(args) < 2:
                raise CypherExecutionError("db.vector.search expects at least index and vector")
            index = args[0]
            vector = args[1]
            k = args[2] if len(args) > 2 else None
            options = args[3] if len(args) > 3 else None
            if not isinstance(index, str):
                raise CypherExecutionError("db.vector.search index must be a string")
            if not isinstance(vector, list):
                raise CypherExecutionError("db.vector.search vector must be a list")
            if options is not None and not isinstance(options, dict):
                raise CypherExecutionError("db.vector.search options must be a map")
            options = options or {}
            candidate_multiplier = options.get("candidate_multiplier")
            reranker_name = options.get("reranker")
            if candidate_multiplier is not None and not isinstance(candidate_multiplier, int):
                raise CypherExecutionError("db.vector.search candidate_multiplier must be an integer")
            if reranker_name is not None and not isinstance(reranker_name, str):
                raise CypherExecutionError("db.vector.search reranker must be a string")
            properties_filter = self._coerce_vector_properties_filter(options.get("properties"))
            rerank_flag = bool(options.get("rerank", False) or reranker_name is not None)
            results = self.db.semantic_search(
                vector,
                k=k,
                index=index,
                filter_labels=options.get("labels"),
                filter_props=properties_filter,
                rerank=rerank_flag,
                reranker=reranker_name,
                candidate_multiplier=candidate_multiplier,
            )
            return [{"node": row["node"], "score": row["score"]} for row in results]

        if lower_name == "db.uri_index.create":
            if len(args) < 1:
                raise CypherExecutionError("db.uri_index.create expects entity ('node' or 'relationship')")
            entity = args[0]
            unique = args[1] if len(args) > 1 else True
            name = args[2] if len(args) > 2 else None
            if not isinstance(entity, str):
                raise CypherExecutionError("db.uri_index.create entity must be a string")
            if unique is not None and not isinstance(unique, bool):
                raise CypherExecutionError("db.uri_index.create unique must be a boolean")
            if name is not None and not isinstance(name, str):
                raise CypherExecutionError("db.uri_index.create name must be a string")
            entity_lower = entity.lower()
            if entity_lower in {"node", "nodes"}:
                index_name = self.db.create_node_uri_index(unique=bool(unique), name=name)
            elif entity_lower in {"relationship", "relationships"}:
                index_name = self.db.create_relationship_uri_index(unique=bool(unique), name=name)
            else:
                raise CypherExecutionError("db.uri_index.create entity must be 'node' or 'relationship'")
            return [{"name": index_name}]

        if lower_name == "apoc.load.jsonarray":
            if len(args) != 1:
                raise CypherExecutionError("apoc.load.jsonArray expects 1 argument")
            source = args[0]
            if not isinstance(source, str):
                raise CypherExecutionError("apoc.load.jsonArray expects a string path or URL")
            data = self._load_json_from_source(source)
            if not isinstance(data, list):
                raise CypherExecutionError("apoc.load.jsonArray expects a JSON array")
            return [{"value": item} for item in data]

        if lower_name == "apoc.load.json":
            if len(args) != 1:
                raise CypherExecutionError("apoc.load.json expects 1 argument")
            source = args[0]
            if not isinstance(source, str):
                raise CypherExecutionError("apoc.load.json expects a string path or URL")
            data = self._load_json_from_source(source)
            if data is None:
                return []
            if isinstance(data, list):
                return [{"value": item} for item in data]
            return [{"value": data}]

        if lower_name == "apoc.load.jsonparams":
            if len(args) not in (3, 4):
                raise CypherExecutionError("apoc.load.jsonParams expects 3 or 4 arguments")
            source = args[0]
            params = args[1]
            headers = args[2]
            options = args[3] if len(args) == 4 else None
            if not isinstance(source, str):
                raise CypherExecutionError("apoc.load.jsonParams expects a string path or URL")
            if not isinstance(params, dict):
                raise CypherExecutionError("apoc.load.jsonParams expects a params map")
            if not isinstance(headers, dict):
                raise CypherExecutionError("apoc.load.jsonParams expects a headers map")
            if options is not None and not isinstance(options, dict):
                raise CypherExecutionError("apoc.load.jsonParams options must be a map")
            url = self._apply_query_params(source, params)
            data = self._load_json_from_source(url, headers=headers, options=options)
            if data is None:
                return []
            if isinstance(data, list):
                return [{"value": item} for item in data]
            return [{"value": data}]

        if lower_name == "apoc.import.json":
            if len(args) not in (1, 2):
                raise CypherExecutionError("apoc.import.json expects 1 or 2 arguments")
            source = args[0]
            options = args[1] if len(args) == 2 else None
            if not isinstance(source, (str, bytes, bytearray)):
                raise CypherExecutionError("apoc.import.json expects a string path/URL or bytes")
            if options is not None and not isinstance(options, dict):
                raise CypherExecutionError("apoc.import.json options must be a map")

            payload = self._load_import_text_with_options(source, options)
            if payload is None:
                return []
            entries = self._parse_import_payload(payload)

            id_field = (options or {}).get("idField", "id")
            labels_field = (options or {}).get("labelsField", "labels")
            props_field = (options or {}).get("propertiesField", "properties")
            rel_type_field = (options or {}).get("relTypeField", "label")
            start_field = (options or {}).get("startField", "start")
            end_field = (options or {}).get("endField", "end")
            type_field = (options or {}).get("typeField", "type")

            node_lookup: dict[Any, int] = {}
            created_nodes = 0
            created_rels = 0

            for entry in entries:
                if not isinstance(entry, dict):
                    raise CypherExecutionError("apoc.import.json entries must be objects")
                entry_type = entry.get(type_field)
                is_relationship = False
                if isinstance(entry_type, str) and entry_type.lower() in ("relationship", "rel", "edge"):
                    is_relationship = True
                if entry_type is None and (start_field in entry or end_field in entry):
                    is_relationship = True

                if not is_relationship:
                    labels = self._normalize_import_labels(entry.get(labels_field))
                    props = self._normalize_import_properties(entry.get(props_field))
                    node = self.db.create_node(labels=labels, properties=props)
                    created_nodes += 1
                    node_id = entry.get(id_field)
                    if node_id is not None:
                        node_lookup[node_id] = node.id
                    continue

                start_ref = self._resolve_import_ref(entry.get(start_field))
                end_ref = self._resolve_import_ref(entry.get(end_field))
                if start_ref is None or end_ref is None:
                    raise CypherExecutionError("apoc.import.json relationships need start/end")
                if start_ref not in node_lookup or end_ref not in node_lookup:
                    raise CypherExecutionError("apoc.import.json relationship references unknown nodes")
                rel_type = entry.get(rel_type_field) or entry.get("type") or "RELATED_TO"
                rel_props = self._normalize_import_properties(entry.get(props_field))
                self.db.create_relationship(
                    source_id=node_lookup[start_ref],
                    target_id=node_lookup[end_ref],
                    rel_type=str(rel_type),
                    properties=rel_props,
                )
                created_rels += 1

            return [{"nodes": created_nodes, "relationships": created_rels}]

        if lower_name == "apoc.load.html":
            if len(args) != 2:
                raise CypherExecutionError("apoc.load.html expects 2 arguments")
            source = args[0]
            selectors = args[1]
            if not isinstance(source, str):
                raise CypherExecutionError("apoc.load.html expects a string path or URL")
            if not isinstance(selectors, dict):
                raise CypherExecutionError("apoc.load.html expects a selector map")
            html = self._load_html_from_source(source)
            root = self._parse_html(html)
            value: dict[str, list[dict[str, Any]]] = {}
            for key, selector in selectors.items():
                if not isinstance(selector, str):
                    raise CypherExecutionError("apoc.load.html selectors must be strings")
                nodes = self._select_html_nodes(root, selector)
                value[key] = [{"text": node.text_content().strip()} for node in nodes]
            return [{"value": value}]

        if lower_name == "apoc.load.xml":
            if len(args) not in (2, 3):
                raise CypherExecutionError("apoc.load.xml expects 2 or 3 arguments")
            source = args[0]
            xpath = args[1]
            options = args[2] if len(args) == 3 else None
            if not isinstance(source, str):
                raise CypherExecutionError("apoc.load.xml expects a string path or URL")
            if not isinstance(xpath, str):
                raise CypherExecutionError("apoc.load.xml expects an XPath string")
            if options is not None and not isinstance(options, dict):
                raise CypherExecutionError("apoc.load.xml options must be a map")
            xml_payload = self._load_xml_from_source(source, options=options)
            if xml_payload is None:
                return []
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(xml_payload)
            except Exception as exc:
                raise CypherExecutionError(f"Failed to parse XML from {source}: {exc}") from exc
            if xpath.startswith("/"):
                xpath = f".{xpath}"
            matches = root.findall(xpath)
            return [{"value": self._element_to_dict(match)} for match in matches]

        if lower_name == "apoc.load.xmlparams":
            if len(args) not in (4, 5):
                raise CypherExecutionError("apoc.load.xmlParams expects 4 or 5 arguments")
            source = args[0]
            xpath = args[1]
            params = args[2]
            headers = args[3]
            options = args[4] if len(args) == 5 else None
            if not isinstance(source, str):
                raise CypherExecutionError("apoc.load.xmlParams expects a string path or URL")
            if not isinstance(xpath, str):
                raise CypherExecutionError("apoc.load.xmlParams expects an XPath string")
            if not isinstance(params, dict):
                raise CypherExecutionError("apoc.load.xmlParams expects a params map")
            if not isinstance(headers, dict):
                raise CypherExecutionError("apoc.load.xmlParams expects a headers map")
            if options is not None and not isinstance(options, dict):
                raise CypherExecutionError("apoc.load.xmlParams options must be a map")
            url = self._apply_query_params(source, params)
            xml_payload = self._load_xml_from_source(url, options=options)
            if xml_payload is None:
                return []
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(xml_payload)
            except Exception as exc:
                raise CypherExecutionError(f"Failed to parse XML from {source}: {exc}") from exc
            if xpath.startswith("/"):
                xpath = f".{xpath}"
            matches = root.findall(xpath)
            return [{"value": self._element_to_dict(match)} for match in matches]

        raise CypherExecutionError(f"Unknown procedure: {name}")

    def _load_json_from_source(
        self,
        source: str,
        headers: dict[str, str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """Load JSON data from a URL or local file path."""
        archive_member = None
        if "!" in source:
            source, archive_member = source.split("!", 1)

        parsed = urllib.parse.urlparse(source)
        options = options or {}
        method = options.get("method") or "GET"
        payload = options.get("payload")
        timeout = options.get("timeout")
        retry = options.get("retry", 0)
        fail_on_error = options.get("failOnError", True)
        options_headers = options.get("headers")
        auth = options.get("auth")
        if timeout is not None and not isinstance(timeout, (int, float)):
            raise CypherExecutionError("apoc.load.jsonParams timeout must be a number")
        if retry is not None and not isinstance(retry, int):
            raise CypherExecutionError("apoc.load.jsonParams retry must be an integer")
        if options_headers is not None and not isinstance(options_headers, dict):
            raise CypherExecutionError("apoc.load.jsonParams headers must be a map")
        if not isinstance(fail_on_error, bool):
            raise CypherExecutionError("apoc.load.jsonParams failOnError must be a boolean")

        if parsed.scheme in ("http", "https"):
            cache_dir = os.getenv("GRAFITO_APOC_CACHE_DIR")
            cache_path = None
            can_cache = (
                method.upper() == "GET"
                and payload is None
                and not headers
                and not options_headers
                and not auth
                and not options.get("params")
            )
            if cache_dir and can_cache and archive_member is None:
                os.makedirs(cache_dir, exist_ok=True)
                url_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
                cache_path = os.path.join(cache_dir, f"{url_hash}.json")
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, encoding="utf-8") as handle:
                        payload = handle.read()
                except Exception as exc:
                    raise CypherExecutionError(
                        f"Failed to read cached JSON {cache_path}: {exc}"
                    ) from exc
            else:
                if archive_member is not None:
                    payload_bytes = self._load_bytes_from_source(
                        source,
                        "JSON",
                        headers=headers,
                        options=options,
                    )
                    if payload_bytes is None:
                        return None
                    payload = self._extract_json_from_archive(
                        payload_bytes,
                        source,
                        archive_member,
                    )
                else:
                    request_headers = {"User-Agent": "GrafitoCypher/0.1"}
                    if headers:
                        request_headers.update(headers)
                    if options_headers:
                        request_headers.update(options_headers)
                    if auth:
                        if isinstance(auth, dict):
                            user = auth.get("user") or auth.get("username")
                            password = auth.get("password") or auth.get("pass")
                            if user is None or password is None:
                                raise CypherExecutionError("apoc.load.jsonParams auth map needs user/password")
                            token = f"{user}:{password}".encode("utf-8")
                        elif isinstance(auth, str):
                            token = auth.encode("utf-8")
                        else:
                            raise CypherExecutionError("apoc.load.jsonParams auth must be a string or map")
                        request_headers["Authorization"] = f"Basic {base64.b64encode(token).decode('ascii')}"

                    data_bytes = None
                    if payload is not None:
                        if isinstance(payload, (dict, list)):
                            data_bytes = orjson.dumps(payload)
                            request_headers.setdefault("Content-Type", "application/json")
                        elif isinstance(payload, str):
                            data_bytes = payload.encode("utf-8")
                        elif isinstance(payload, bytes):
                            data_bytes = payload
                        else:
                            raise CypherExecutionError("apoc.load.jsonParams payload must be string, bytes, list, or map")

                    attempts = retry + 1 if retry is not None else 1
                    last_exc: Exception | None = None
                    for _ in range(attempts):
                        try:
                            request = urllib.request.Request(
                                source,
                                headers=request_headers,
                                data=data_bytes,
                                method=method.upper(),
                            )
                            with urllib.request.urlopen(request, timeout=timeout) as handle:
                                payload = handle.read().decode("utf-8")
                            last_exc = None
                            break
                        except Exception as exc:
                            last_exc = exc
                    if last_exc is not None:
                        if fail_on_error:
                            raise CypherExecutionError(
                                f"Failed to load JSON from {source}: {last_exc}"
                            ) from last_exc
                        return None

                if cache_path:
                    try:
                        with open(cache_path, "w", encoding="utf-8") as handle:
                            handle.write(payload)
                    except Exception:
                        pass
        else:
            path = source
            if parsed.scheme == "file":
                path = urllib.request.url2pathname(parsed.path)
            if not os.path.exists(path):
                raise CypherExecutionError(f"JSON file not found: {path}")
            try:
                if archive_member is not None:
                    with open(path, "rb") as handle:
                        payload_bytes = handle.read()
                    payload = self._extract_json_from_archive(
                        payload_bytes,
                        path,
                        archive_member,
                    )
                else:
                    with open(path, encoding="utf-8") as handle:
                        payload = handle.read()
            except Exception as exc:
                raise CypherExecutionError(f"Failed to read JSON file {path}: {exc}") from exc
        if payload is None:
            return None
        try:
            data = orjson.loads(payload)
        except orjson.JSONDecodeError as exc:
            cleaned = payload.lstrip("\ufeff")
            cleaned = re.sub(r",\s*(\]|\})", r"\1", cleaned)
            try:
                data = orjson.loads(cleaned)
            except orjson.JSONDecodeError as exc2:
                raise CypherExecutionError(
                    f"Invalid JSON payload from {source}: {exc}"
                ) from exc2
        return data

    def _load_html_from_source(self, source: str) -> str:
        """Load HTML content from a URL or local file path."""
        parsed = urllib.parse.urlparse(source)
        if parsed.scheme in ("http", "https"):
            try:
                request = urllib.request.Request(
                    source,
                    headers={"User-Agent": "GrafitoCypher/0.1"},
                )
                with urllib.request.urlopen(request) as handle:
                    return handle.read().decode("utf-8")
            except Exception as exc:
                raise CypherExecutionError(f"Failed to load HTML from {source}: {exc}") from exc
        path = source
        if parsed.scheme == "file":
            path = urllib.request.url2pathname(parsed.path)
        if not os.path.exists(path):
            raise CypherExecutionError(f"HTML file not found: {path}")
        try:
            with open(path, encoding="utf-8") as handle:
                return handle.read()
        except Exception as exc:
            raise CypherExecutionError(f"Failed to read HTML file {path}: {exc}") from exc

    def _load_bytes_from_source(
        self,
        source: str,
        kind: str,
        headers: dict[str, str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> bytes | None:
        """Load raw bytes from a URL or local file path."""
        parsed = urllib.parse.urlparse(source)
        options = options or {}
        method = options.get("method") or "GET"
        payload = options.get("payload")
        timeout = options.get("timeout")
        retry = options.get("retry", 0)
        fail_on_error = options.get("failOnError", True)
        options_headers = options.get("headers")
        auth = options.get("auth")
        if timeout is not None and not isinstance(timeout, (int, float)):
            raise CypherExecutionError(f"apoc.load.{kind.lower()}Params timeout must be a number")
        if retry is not None and not isinstance(retry, int):
            raise CypherExecutionError(f"apoc.load.{kind.lower()}Params retry must be an integer")
        if options_headers is not None and not isinstance(options_headers, dict):
            raise CypherExecutionError(f"apoc.load.{kind.lower()}Params headers must be a map")
        if not isinstance(fail_on_error, bool):
            raise CypherExecutionError(f"apoc.load.{kind.lower()}Params failOnError must be a boolean")

        if parsed.scheme in ("http", "https"):
            request_headers = {"User-Agent": "GrafitoCypher/0.1"}
            if headers:
                request_headers.update(headers)
            if options_headers:
                request_headers.update(options_headers)
            if auth:
                if isinstance(auth, dict):
                    user = auth.get("user") or auth.get("username")
                    password = auth.get("password") or auth.get("pass")
                    if user is None or password is None:
                        raise CypherExecutionError(
                            f"apoc.load.{kind.lower()}Params auth map needs user/password"
                        )
                    token = f"{user}:{password}".encode("utf-8")
                elif isinstance(auth, str):
                    token = auth.encode("utf-8")
                else:
                    raise CypherExecutionError(
                        f"apoc.load.{kind.lower()}Params auth must be a string or map"
                    )
                request_headers["Authorization"] = f"Basic {base64.b64encode(token).decode('ascii')}"

            data_bytes = None
            if payload is not None:
                if isinstance(payload, (dict, list)):
                    data_bytes = orjson.dumps(payload)
                    request_headers.setdefault("Content-Type", "application/json")
                elif isinstance(payload, str):
                    data_bytes = payload.encode("utf-8")
                elif isinstance(payload, bytes):
                    data_bytes = payload
                else:
                    raise CypherExecutionError(
                        f"apoc.load.{kind.lower()}Params payload must be string, bytes, list, or map"
                    )

            attempts = retry + 1 if retry is not None else 1
            last_exc: Exception | None = None
            for _ in range(attempts):
                try:
                    request = urllib.request.Request(
                        source,
                        headers=request_headers,
                        data=data_bytes,
                        method=method.upper(),
                    )
                    with urllib.request.urlopen(request, timeout=timeout) as handle:
                        return handle.read()
                except Exception as exc:
                    last_exc = exc
            if last_exc is not None:
                if fail_on_error:
                    raise CypherExecutionError(
                        f"Failed to load {kind} from {source}: {last_exc}"
                    ) from last_exc
                return None
            return None
        path = source
        if parsed.scheme == "file":
            path = urllib.request.url2pathname(parsed.path)
        if not os.path.exists(path):
            raise CypherExecutionError(f"{kind} file not found: {path}")
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except Exception as exc:
            raise CypherExecutionError(f"Failed to read {kind} file {path}: {exc}") from exc

    def _load_xml_from_source(self, source: str, options: dict[str, Any] | None = None) -> str | None:
        """Load XML content from a URL or local file path with optional compression."""
        payload = self._load_bytes_from_source(source, "XML", options=options)
        if payload is None:
            return None
        compression = None
        archive_path = None
        if options:
            compression = options.get("compression")
            archive_path = options.get("path") or options.get("fileName")
        if compression is None:
            lower_source = source.lower()
            if lower_source.endswith(".gz"):
                compression = "gzip"
            elif lower_source.endswith(".bz2"):
                compression = "bz2"
            elif lower_source.endswith(".xz") or lower_source.endswith(".lzma"):
                compression = "xz"
            elif lower_source.endswith(".zip"):
                compression = "zip"

        if compression:
            compression = str(compression).lower()
            if compression in ("gzip", "gz"):
                import gzip

                payload = gzip.decompress(payload)
            elif compression in ("bz2", "bzip2"):
                import bz2

                payload = bz2.decompress(payload)
            elif compression in ("xz", "lzma"):
                import lzma

                payload = lzma.decompress(payload)
            elif compression == "zip":
                import zipfile

                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    members = archive.namelist()
                    if archive_path:
                        if archive_path not in members:
                            raise CypherExecutionError(
                                f"XML entry not found in zip archive: {archive_path}"
                            )
                        target = archive_path
                    else:
                        xml_members = [name for name in members if name.lower().endswith(".xml")]
                        target = xml_members[0] if xml_members else members[0]
                    payload = archive.read(target)
            else:
                raise CypherExecutionError(f"Unsupported XML compression: {compression}")

        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CypherExecutionError(f"Failed to decode XML from {source}: {exc}") from exc

    def _element_to_dict(self, element: Any) -> dict[str, Any]:
        """Convert an XML element to a nested dict structure."""
        result: dict[str, Any] = {"_tag": element.tag}
        if element.attrib:
            result["_attributes"] = dict(element.attrib)
        text = (element.text or "").strip()
        if text:
            result["_text"] = text
        for child in list(element):
            child_value = self._element_to_dict(child)
            key = child.tag
            if key in result:
                existing = result[key]
                if isinstance(existing, list):
                    existing.append(child_value)
                else:
                    result[key] = [existing, child_value]
            else:
                result[key] = child_value
        return result

    def _parse_html(self, html: str) -> _HtmlNode:
        """Parse HTML into a simple node tree."""
        parser = _HtmlTreeBuilder()
        parser.feed(html)
        parser.close()
        return parser.root

    def _parse_html_selector_segment(self, segment: str) -> tuple[str | None, list[str], int | None]:
        """Parse a minimal selector segment: tag(.class)*(:eq(n))?"""
        eq = None
        base = segment
        if ":eq(" in segment:
            base, eq_part = segment.split(":eq(", 1)
            if not eq_part.endswith(")"):
                raise CypherExecutionError(f"Unsupported HTML selector segment: {segment}")
            try:
                eq = int(eq_part[:-1])
            except ValueError as exc:
                raise CypherExecutionError(f"Unsupported HTML selector segment: {segment}") from exc

        base = base.strip()
        if not base:
            raise CypherExecutionError(f"Unsupported HTML selector segment: {segment}")

        parts = base.split(".")
        tag = parts[0] or None
        classes = [part for part in parts[1:] if part]

        name_pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
        if tag and not name_pattern.match(tag):
            raise CypherExecutionError(f"Unsupported HTML selector segment: {segment}")
        for cls in classes:
            if not name_pattern.match(cls):
                raise CypherExecutionError(f"Unsupported HTML selector segment: {segment}")

        return tag.lower() if tag else None, classes, eq

    def _select_html_nodes(self, root: _HtmlNode, selector: str) -> list[_HtmlNode]:
        """Select nodes using a minimal descendant-selector engine."""
        segments = [segment for segment in selector.split() if segment]
        if not segments:
            return []

        current = [root]
        for segment in segments:
            tag, classes, eq = self._parse_html_selector_segment(segment)
            matches: list[_HtmlNode] = []
            for base in current:
                stack = list(reversed(base.children))
                while stack:
                    node = stack.pop()
                    stack.extend(reversed(node.children))
                    if not self._html_node_matches(node, tag, classes, eq):
                        continue
                    matches.append(node)
            current = matches
        return current

    def _html_node_matches(
        self,
        node: _HtmlNode,
        tag: str | None,
        classes: list[str],
        eq: int | None,
    ) -> bool:
        """Match a node against a selector segment."""
        if tag and node.tag != tag:
            return False
        if classes:
            node_classes = node.classes()
            if any(cls not in node_classes for cls in classes):
                return False
        if eq is not None:
            if node.parent is None:
                return False
            siblings = [child for child in node.parent.children if child.tag == node.tag]
            try:
                index = siblings.index(node)
            except ValueError:
                return False
            if index != eq:
                return False
        return True

    def _coerce_vector_properties_filter(self, props: Any) -> Any:
        """Coerce Cypher map filters into PropertyFilter/PropertyFilterGroup."""
        if props is None:
            return None
        if isinstance(props, PropertyFilterGroup):
            return props
        if not isinstance(props, dict):
            raise CypherExecutionError("db.vector.search properties must be a map")

        logical_key = None
        if "$or" in props or "or" in props:
            logical_key = "$or" if "$or" in props else "or"
            operator = "OR"
        elif "$and" in props or "and" in props:
            logical_key = "$and" if "$and" in props else "and"
            operator = "AND"

        if logical_key is not None:
            if len(props) != 1:
                raise CypherExecutionError(
                    "Logical properties filter must only contain the operator key"
                )
            filters = props[logical_key]
            if not isinstance(filters, list):
                raise CypherExecutionError("Logical properties filter must be a list")
            dict_filters = [self._coerce_properties_map(item) for item in filters]
            return PropertyFilterGroup(operator, *dict_filters)

        return self._coerce_properties_map(props)

    def _apply_query_params(self, source: str, params: dict[str, Any]) -> str:
        """Apply query parameters to a URL."""
        parsed = urllib.parse.urlparse(source)
        if parsed.scheme not in ("http", "https"):
            return source
        query_items = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                for item in value:
                    query_items.append((key, item))
            else:
                query_items.append((key, value))
        query = urllib.parse.urlencode(query_items, doseq=True)
        return urllib.parse.urlunparse(parsed._replace(query=query))

    def _extract_json_from_archive(self, payload: bytes, source: str, member: str) -> str:
        """Extract a JSON payload from a tar archive."""
        import tarfile

        lower_source = source.lower()
        if lower_source.endswith(".tgz") or lower_source.endswith(".tar.gz"):
            mode = "r:gz"
        elif lower_source.endswith(".tar.bz2") or lower_source.endswith(".tbz2"):
            mode = "r:bz2"
        elif lower_source.endswith(".tar.xz") or lower_source.endswith(".txz"):
            mode = "r:xz"
        elif lower_source.endswith(".tar"):
            mode = "r:"
        else:
            raise CypherExecutionError(
                f"Unsupported JSON archive type for {source}; expected .tar, .tgz, .tar.gz, .tar.bz2, or .tar.xz"
            )

        normalized_member = member.lstrip("./")
        with tarfile.open(fileobj=io.BytesIO(payload), mode=mode) as archive:
            for item in archive.getmembers():
                if item.isdir():
                    continue
                name = item.name.lstrip("./")
                if name == normalized_member:
                    handle = archive.extractfile(item)
                    if handle is None:
                        break
                    data = handle.read()
                    try:
                        return data.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise CypherExecutionError(
                            f"Failed to decode JSON entry {member} from {source}: {exc}"
                        ) from exc
        raise CypherExecutionError(f"JSON entry not found in archive {source}: {member}")

    def _load_import_text(self, source: str) -> str | None:
        """Load raw text for JSON/JSONL import, supporting tar archives."""
        return self._load_import_text_with_options(source, None)

    def _load_import_text_with_options(self, source: Any, options: dict[str, Any] | None) -> str | None:
        """Load import payload from string/bytes with optional decompression."""
        if isinstance(source, (bytes, bytearray)):
            payload_bytes = bytes(source)
            return self._decode_import_payload(payload_bytes, "<memory>", options)

        if not isinstance(source, str):
            raise CypherExecutionError("apoc.import.json expects a string path/URL or bytes")

        archive_member = None
        if "!" in source:
            source, archive_member = source.split("!", 1)

        payload_bytes = self._load_bytes_from_source(source, "JSON")
        if payload_bytes is None:
            return None
        if archive_member is not None:
            return self._extract_json_from_archive(payload_bytes, source, archive_member)
        return self._decode_import_payload(payload_bytes, source, options)

    def _decode_import_payload(
        self,
        payload: bytes,
        source: str,
        options: dict[str, Any] | None,
    ) -> str:
        """Decode import payload, applying optional compression."""
        compression = None
        if options:
            compression = options.get("compression")
        if compression:
            payload = self._decompress_payload(payload, compression, options or {}, source)
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CypherExecutionError(f"Failed to decode JSON from {source}: {exc}") from exc

    def _decompress_payload(
        self,
        payload: bytes,
        compression: str,
        options: dict[str, Any],
        source: str,
    ) -> bytes:
        """Decompress raw payload bytes based on compression option."""
        compression = str(compression).upper()
        if compression == "DEFLATE":
            import zlib

            return zlib.decompress(payload)
        if compression in ("GZIP", "GZ"):
            import gzip

            return gzip.decompress(payload)
        if compression in ("BZ2", "BZIP2"):
            import bz2

            return bz2.decompress(payload)
        if compression in ("XZ", "LZMA"):
            import lzma

            return lzma.decompress(payload)
        if compression == "ZIP":
            import zipfile

            path = options.get("path")
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                members = archive.namelist()
                target = path or (members[0] if members else None)
                if target is None or target not in members:
                    raise CypherExecutionError(
                        f"JSON entry not found in zip archive {source}: {path}"
                    )
                return archive.read(target)
        raise CypherExecutionError(f"Unsupported compression type: {compression}")

    def _parse_import_payload(self, payload: str) -> list[dict[str, Any]]:
        """Parse JSON or JSONL payload into a list of entries."""
        try:
            data = orjson.loads(payload)
        except orjson.JSONDecodeError:
            entries: list[dict[str, Any]] = []
            for line in payload.splitlines():
                line = line.strip()
                if not line:
                    continue
                entries.append(orjson.loads(line))
            return entries

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            nodes = data.get("nodes") or []
            rels = data.get("relationships") or data.get("rels") or []
            if nodes or rels:
                return list(nodes) + list(rels)
            return [data]
        raise CypherExecutionError("apoc.import.json expects JSON object, array, or JSONL")

    def _normalize_import_labels(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
        raise CypherExecutionError("apoc.import.json labels must be a string or list")

    def _normalize_import_properties(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise CypherExecutionError("apoc.import.json properties must be a map")

    def _resolve_import_ref(self, value: Any) -> Any:
        if isinstance(value, dict):
            return value.get("id") or value.get("identity")
        return value

    def _coerce_properties_map(self, props: Any) -> dict:
        if not isinstance(props, dict):
            raise CypherExecutionError("Properties filter entries must be maps")
        coerced = {}
        for key, value in props.items():
            if isinstance(value, dict):
                coerced[key] = self._coerce_property_operator(value)
            else:
                coerced[key] = value
        return coerced

    def _coerce_property_operator(self, spec: dict) -> PropertyFilter:
        operator_map = {
            "gt": PropertyFilter.gt,
            "$gt": PropertyFilter.gt,
            "lt": PropertyFilter.lt,
            "$lt": PropertyFilter.lt,
            "gte": PropertyFilter.gte,
            "$gte": PropertyFilter.gte,
            "lte": PropertyFilter.lte,
            "$lte": PropertyFilter.lte,
            "ne": PropertyFilter.ne,
            "$ne": PropertyFilter.ne,
            "between": PropertyFilter.between,
            "$between": PropertyFilter.between,
            "contains": PropertyFilter.contains,
            "$contains": PropertyFilter.contains,
            "starts_with": PropertyFilter.starts_with,
            "$starts_with": PropertyFilter.starts_with,
            "ends_with": PropertyFilter.ends_with,
            "$ends_with": PropertyFilter.ends_with,
            "regex": PropertyFilter.regex,
            "$regex": PropertyFilter.regex,
        }

        case_sensitive = spec.get("case_sensitive")
        if case_sensitive is not None and not isinstance(case_sensitive, bool):
            raise CypherExecutionError("case_sensitive must be a boolean")

        operator_keys = [key for key in spec.keys() if key != "case_sensitive"]
        if len(operator_keys) != 1:
            raise CypherExecutionError("Property filter map must contain a single operator")

        op_key = operator_keys[0]
        if op_key not in operator_map:
            raise CypherExecutionError(f"Unsupported property filter operator: {op_key}")

        value = spec[op_key]
        if op_key in ("between", "$between"):
            if not isinstance(value, list) or len(value) != 2:
                raise CypherExecutionError("between expects a two-item list")
            return operator_map[op_key](value[0], value[1])

        if op_key in ("contains", "$contains", "starts_with", "$starts_with", "ends_with", "$ends_with"):
            if case_sensitive is None:
                return operator_map[op_key](value)
            return operator_map[op_key](value, case_sensitive=case_sensitive)

        return operator_map[op_key](value)

    def _execute_query_with_context(self, query: Query, initial_results: list[dict]) -> list[dict]:
        """Execute a query with initial results as context."""
        if query.union_clauses:
            return self._execute_union(query)

        if query.clauses:
            return self._execute_multi_clause(query.clauses, initial_results=initial_results)

        if isinstance(query.clause, WithClause):
            return self._execute_with(query.clause, initial_results)

        if isinstance(query.clause, SubqueryClause):
            return self._execute_subquery(query.clause, initial_results)
        if isinstance(query.clause, ProcedureCallClause):
            return self._execute_procedure_call(query.clause, initial_results)

        if isinstance(query.clause, UnwindClause):
            return self._execute_unwind(query.clause, initial_results)
        if isinstance(query.clause, LoadCsvClause):
            return self._execute_load_csv(query.clause, initial_results)

        if isinstance(query.clause, MatchClause):
            return self._execute_match(query.clause, context=initial_results)
        elif isinstance(query.clause, CreateClause):
            return self._execute_create(query.clause)
        elif isinstance(query.clause, MergeClause):
            return self._execute_merge(query.clause)
        else:
            raise CypherExecutionError(f"Unknown clause type: {type(query.clause)}")

    def _execute_unwind(self, clause: UnwindClause, input_results: list[dict]) -> list[dict]:
        """Execute UNWIND clause to expand a list into rows."""
        if not input_results:
            input_results = [{}]

        results = []
        for match in input_results:
            evaluator = self._make_evaluator(match)
            value = evaluator.evaluate(clause.list_expr)
            if value is None:
                continue
            if not isinstance(value, (list, tuple)):
                raise CypherExecutionError("UNWIND expects a list expression")
            for item in value:
                row = match.copy()
                row[clause.variable] = item
                results.append(row)
        return results

    def _execute_create(self, clause: CreateClause, context: list[dict] | None = None) -> list[dict]:
        """Execute CREATE clause.

        Supports:
        - Single node: CREATE (n:Person {name: 'Alice'})
        - Multiple nodes: CREATE (a:Person), (b:Person)
        - Nodes with relationships: CREATE (a:Person)-[r:KNOWS]->(b:Person)

        Args:
            clause: CreateClause AST node

        Returns:
            List with created entity info
        """
        if context:
            results = []
            for row in context:
                row_result = row.copy()
                for pattern in clause.patterns:
                    self._apply_create_pattern_with_context(pattern, row_result)
                results.append(row_result)
            return results

        results = []
        for pattern in clause.patterns:
            if len(pattern.elements) == 1 and pattern.elements[0].relationship is None:
                # Simple node creation: CREATE (n:Person)
                elem = pattern.elements[0]
                node_pattern = elem.node
                properties = self._evaluate_properties(node_pattern.properties, None)
                node = self.db.create_node(
                    labels=node_pattern.labels,
                    properties=properties
                )

                # Build result
                result = {}
                if node_pattern.variable:
                    result[node_pattern.variable] = node.to_dict()
                results.append(result if result else {'created': node.to_dict()})
            else:
                # Pattern with relationships: CREATE (a)-[r]->(b)
                result = self._create_pattern_with_relationships(pattern)
                results.append(result)

        return results

    def _create_pattern_with_relationships(self, pattern: Pattern, return_dicts: bool = True) -> dict:
        """Create nodes and relationships from a pattern.

        Args:
            pattern: Pattern AST node

        Returns:
            Dictionary with created nodes and relationships
        """
        if len(pattern.elements) != 2:
            raise CypherExecutionError(
                "Only single-hop relationship patterns supported in CREATE (e.g., (a)-[r]->(b))"
            )

        source_elem = pattern.elements[0]
        target_elem = pattern.elements[1]
        rel_pattern = source_elem.relationship

        if rel_pattern is None:
            raise CypherExecutionError("Invalid relationship pattern in CREATE")
        if rel_pattern.min_hops != 1 or rel_pattern.max_hops != 1:
            raise CypherExecutionError("Variable-length relationships are not supported in CREATE")

        # Create source node
        source_properties = self._evaluate_properties(source_elem.node.properties, None)
        source_node = self.db.create_node(
            labels=source_elem.node.labels,
            properties=source_properties
        )

        # Create target node
        target_properties = self._evaluate_properties(target_elem.node.properties, None)
        target_node = self.db.create_node(
            labels=target_elem.node.labels,
            properties=target_properties
        )

        # Create relationship
        # For CREATE, we ignore direction and always create outgoing from source to target
        rel_type = rel_pattern.rel_type if rel_pattern.rel_type else "RELATED_TO"
        rel_properties = self._evaluate_properties(rel_pattern.properties, None)
        relationship = self.db.create_relationship(
            source_id=source_node.id,
            target_id=target_node.id,
            rel_type=rel_type,
            properties=rel_properties
        )

        # Build result
        result = {}
        if source_elem.node.variable:
            result[source_elem.node.variable] = source_node.to_dict() if return_dicts else source_node
        if rel_pattern.variable:
            result[rel_pattern.variable] = relationship.to_dict() if return_dicts else relationship
        if target_elem.node.variable:
            result[target_elem.node.variable] = target_node.to_dict() if return_dicts else target_node

        return result

    def _apply_create_pattern_with_context(self, pattern: Pattern, row: dict) -> None:
        """Create nodes/relationships honoring existing bindings in row."""
        if len(pattern.elements) == 1 and pattern.elements[0].relationship is None:
            elem = pattern.elements[0]
            node_pattern = elem.node
            if node_pattern.variable and node_pattern.variable in row:
                return
            properties = self._evaluate_properties(node_pattern.properties, row)
            node = self.db.create_node(
                labels=node_pattern.labels,
                properties=properties
            )
            if node_pattern.variable:
                row[node_pattern.variable] = node
            return

        if len(pattern.elements) != 2:
            raise CypherExecutionError(
                "Only single-hop relationship patterns supported in CREATE (e.g., (a)-[r]->(b))"
            )

        source_elem = pattern.elements[0]
        target_elem = pattern.elements[1]
        rel_pattern = source_elem.relationship

        if rel_pattern is None:
            raise CypherExecutionError("Invalid relationship pattern in CREATE")
        if rel_pattern.min_hops != 1 or rel_pattern.max_hops != 1:
            raise CypherExecutionError("Variable-length relationships are not supported in CREATE")

        source_node = None
        if source_elem.node.variable:
            source_node = self._resolve_bound_node(row.get(source_elem.node.variable))
        if source_node is None:
            source_properties = self._evaluate_properties(source_elem.node.properties, row)
            source_node = self.db.create_node(
                labels=source_elem.node.labels,
                properties=source_properties
            )
            if source_elem.node.variable:
                row[source_elem.node.variable] = source_node

        target_node = None
        if target_elem.node.variable:
            target_node = self._resolve_bound_node(row.get(target_elem.node.variable))
        if target_node is None:
            target_properties = self._evaluate_properties(target_elem.node.properties, row)
            target_node = self.db.create_node(
                labels=target_elem.node.labels,
                properties=target_properties
            )
            if target_elem.node.variable:
                row[target_elem.node.variable] = target_node

        rel_type = rel_pattern.rel_type if rel_pattern.rel_type else "RELATED_TO"
        rel_properties = self._evaluate_properties(rel_pattern.properties, row)
        relationship = self.db.create_relationship(
            source_id=source_node.id,
            target_id=target_node.id,
            rel_type=rel_type,
            properties=rel_properties
        )

        if rel_pattern.variable:
            row[rel_pattern.variable] = relationship
        if pattern.variable:
            row[pattern.variable] = Path(nodes=[source_node, target_node], relationships=[relationship])

    def _resolve_bound_node(self, value: Any) -> Node | None:
        """Resolve a bound node from a context value."""
        if isinstance(value, Node):
            return value
        if isinstance(value, dict) and 'id' in value:
            return self.db.get_node(value['id'])
        return None

    def _execute_merge(
        self,
        clause: MergeClause,
        context: list[dict] | None = None
    ) -> list[dict]:
        """Execute MERGE clause (find or create pattern).

        MERGE is like an "upsert":
        1. Try to MATCH the pattern
        2. If found: run ON MATCH SET (if present)
        3. If not found: CREATE the pattern and run ON CREATE SET (if present)

        Args:
            clause: MergeClause AST node

        Returns:
            List with node/relationship info
        """
        def build_match_result(matches: list[dict], return_dicts: bool) -> dict:
            result: dict[str, Any] = {}
            for match in matches:
                for var_name, obj in match.items():
                    if return_dicts and hasattr(obj, 'to_dict'):
                        result[var_name] = obj.to_dict()
                    else:
                        result[var_name] = obj
                break
            return result

        def create_pattern_result(
            pattern: Pattern,
            row_context: dict[str, Any] | None,
            return_dicts: bool,
        ) -> dict:
            if len(pattern.elements) == 1 and pattern.elements[0].relationship is None:
                elem = pattern.elements[0]
                node_pattern = elem.node
                properties = self._evaluate_properties(node_pattern.properties, row_context)
                node = self.db.create_node(
                    labels=node_pattern.labels,
                    properties=properties
                )

                if clause.on_create_set:
                    match = {node_pattern.variable: node} if node_pattern.variable else {}
                    self._execute_set([match], clause.on_create_set)
                    node = self.db.get_node(node.id)

                result = {}
                if node_pattern.variable:
                    result[node_pattern.variable] = node.to_dict() if return_dicts else node
                return result if result else {'created': node.to_dict() if return_dicts else node}

            if row_context:
                source_elem = pattern.elements[0]
                target_elem = pattern.elements[1]
                rel_pattern = source_elem.relationship
                if rel_pattern is None:
                    raise CypherExecutionError("Invalid relationship pattern in MERGE")

                source_node = None
                if source_elem.node.variable and source_elem.node.variable in row_context:
                    source_node = self._resolve_bound_node(row_context.get(source_elem.node.variable))
                if source_node is None:
                    source_props = self._evaluate_properties(source_elem.node.properties, row_context)
                    source_node = self.db.create_node(
                        labels=source_elem.node.labels,
                        properties=source_props
                    )

                target_node = None
                if target_elem.node.variable and target_elem.node.variable in row_context:
                    target_node = self._resolve_bound_node(row_context.get(target_elem.node.variable))
                if target_node is None:
                    target_props = self._evaluate_properties(target_elem.node.properties, row_context)
                    target_node = self.db.create_node(
                        labels=target_elem.node.labels,
                        properties=target_props
                    )

                rel_type = rel_pattern.rel_type if rel_pattern.rel_type else "RELATED_TO"
                rel_props = self._evaluate_properties(rel_pattern.properties, row_context)

                existing_rels: list[Relationship] = []
                if rel_pattern.direction == "outgoing":
                    existing_rels = self.db.match_relationships(
                        source_id=source_node.id,
                        target_id=target_node.id,
                        rel_type=rel_type,
                        properties=rel_props if rel_props else None
                    )
                elif rel_pattern.direction == "incoming":
                    existing_rels = self.db.match_relationships(
                        source_id=target_node.id,
                        target_id=source_node.id,
                        rel_type=rel_type,
                        properties=rel_props if rel_props else None
                    )
                else:
                    existing_rels = self.db.match_relationships(
                        source_id=source_node.id,
                        target_id=target_node.id,
                        rel_type=rel_type,
                        properties=rel_props if rel_props else None
                    )
                    if not existing_rels:
                        existing_rels = self.db.match_relationships(
                            source_id=target_node.id,
                            target_id=source_node.id,
                            rel_type=rel_type,
                            properties=rel_props if rel_props else None
                        )

                if existing_rels:
                    rel = existing_rels[0]
                else:
                    rel = self.db.create_relationship(
                        source_id=source_node.id,
                        target_id=target_node.id,
                        rel_type=rel_type,
                        properties=rel_props
                    )

                created_result = {}
                if source_elem.node.variable:
                    created_result[source_elem.node.variable] = (
                        source_node.to_dict() if return_dicts else source_node
                    )
                if rel_pattern.variable:
                    created_result[rel_pattern.variable] = rel.to_dict() if return_dicts else rel
                if target_elem.node.variable:
                    created_result[target_elem.node.variable] = (
                        target_node.to_dict() if return_dicts else target_node
                    )
            else:
                created_result = self._create_pattern_with_relationships(
                    pattern,
                    return_dicts=return_dicts
                )

            if clause.on_create_set:
                match = {}
                for var_name, entity in created_result.items():
                    if hasattr(entity, 'labels'):
                        match[var_name] = entity
                    elif hasattr(entity, 'source_id'):
                        match[var_name] = entity
                    elif isinstance(entity, dict) and 'labels' in entity:
                        match[var_name] = self.db.get_node(entity['id'])
                    elif isinstance(entity, dict) and 'type' in entity:
                        match[var_name] = self.db.get_relationship(entity['id'])

                if match:
                    self._execute_set([match], clause.on_create_set)

                    if return_dicts:
                        for var_name, entity in match.items():
                            if hasattr(entity, 'to_dict'):
                                created_result[var_name] = entity.to_dict()

            return created_result

        if context:
            results = []
            for row in context:
                row_result = row.copy()
                for pattern in clause.patterns:
                    matches = [
                        match for match in self._match_pattern(pattern, context=row_result)
                        if self._pattern_bindings_match(row_result, match)
                    ]

                    if matches:
                        if clause.on_match_set:
                            self._execute_set(matches, clause.on_match_set)
                        merge_result = build_match_result(matches, return_dicts=False)
                    else:
                        merge_result = create_pattern_result(
                            pattern,
                            row_context=row_result,
                            return_dicts=False
                        )
                    row_result.update(merge_result)
                results.append(row_result)
            return results

        results = []
        for pattern in clause.patterns:
            matches = self._match_pattern(pattern)

            if matches:
                if clause.on_match_set:
                    self._execute_set(matches, clause.on_match_set)
                results.append(build_match_result(matches, return_dicts=True))
            else:
                results.append(create_pattern_result(pattern, row_context=None, return_dicts=True))

        return results

    def _execute_match(self, clause: MatchClause, context: list[dict] = None) -> list[dict]:
        """Execute MATCH clause with optional WHERE, DELETE, SET, RETURN, ORDER BY, SKIP, LIMIT.

        Args:
            clause: MatchClause AST node
            context: Optional context from previous WITH clause (not yet fully implemented)

        Returns:
            List of result dictionaries
        """
        pattern_vars = self._collect_pattern_variables(clause.patterns) if clause.optional else set()
        input_rows = context if context else [{}]
        matches = []

        for row in input_rows:
            property_filters = self._extract_property_filters(clause.where_clause, context=row)
            row_matches = [row]
            for pattern in clause.patterns:
                pattern_matches = []
                for partial in row_matches:
                    for match in self._match_pattern(pattern, property_filters, context=partial):
                        if not self._pattern_bindings_match(partial, match):
                            continue
                        merged = partial.copy()
                        merged.update(match)
                        pattern_matches.append(merged)
                row_matches = pattern_matches
                if not row_matches:
                    break

            if clause.where_clause:
                row_matches = self._filter_where(row_matches, clause.where_clause)

            if row_matches:
                matches.extend(row_matches)
            elif clause.optional:
                fallback = row.copy()
                for var in pattern_vars:
                    fallback.setdefault(var, None)
                matches.append(fallback)

        if clause.optional and not matches and not context:
            matches = [{var: None for var in pattern_vars}]

        # Apply DELETE if present (mutating operation)
        if clause.delete_clause:
            return self._execute_delete(matches, clause.delete_clause)

        # Apply SET if present (mutating operation)
        if clause.set_clause:
            self._execute_set(matches, clause.set_clause)

        # Apply REMOVE if present (mutating operation)
        if clause.remove_clause:
            self._execute_remove(matches, clause.remove_clause)

        # When RETURN aggregates, grouping happens during projection, so ORDER BY
        # / SKIP must run on the grouped rows (and may reference aggregate aliases).
        return_aggregates = bool(clause.return_clause) and any(
            isinstance(item.expression, FunctionCall) for item in clause.return_clause.items
        )

        # Apply ORDER BY BEFORE RETURN (needs access to Node objects)
        if clause.order_by_clause and not return_aggregates:
            matches = self._apply_order_by(matches, clause.order_by_clause)

        # Apply SKIP (before LIMIT and RETURN)
        if clause.skip_clause and not return_aggregates:
            matches = matches[clause.skip_clause.count:]

        # Apply LIMIT (on raw matches if no RETURN, or will be applied after RETURN)
        if clause.limit_clause and not clause.return_clause:
            matches = matches[:clause.limit_clause.count]

        # Apply RETURN projection if present
        if clause.return_clause:
            matches = self._apply_return(matches, clause.return_clause)
            # ORDER BY / SKIP run after the aggregated projection so they can
            # sort/skip by grouping keys and aggregate aliases.
            if return_aggregates:
                if clause.order_by_clause:
                    matches = self._apply_order_by(matches, clause.order_by_clause)
                if clause.skip_clause:
                    matches = matches[clause.skip_clause.count:]
            # Apply LIMIT after RETURN if present
            if clause.limit_clause:
                matches = matches[:clause.limit_clause.count]

        return matches

    def _match_pattern(
        self,
        pattern: Pattern,
        property_filters: dict[str, dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None
    ) -> list[dict]:
        """Match a pattern and return variable bindings.

        Args:
            pattern: Pattern AST node

        Returns:
            List of dictionaries mapping variables to matched entities
        """
        if isinstance(pattern, PatternFunction):
            return self._match_pattern_function(pattern)

        if len(pattern.elements) == 1 and pattern.elements[0].relationship is None:
            # Simple node pattern: (n:Label {props})
            return self._match_single_node(
                pattern.elements[0],
                path_variable=pattern.variable,
                property_filters=property_filters,
                context=context
            )
        else:
            # Relationship pattern: (a)-[r]->(b)
            return self._match_relationship_pattern(
                pattern,
                path_variable=pattern.variable,
                property_filters=property_filters,
                context=context
            )

    def _collect_pattern_variables(self, patterns: list[Pattern]) -> set[str]:
        """Collect variable names from patterns."""
        variables: set[str] = set()
        for pattern in patterns:
            if isinstance(pattern, PatternFunction):
                if pattern.variable:
                    variables.add(pattern.variable)
                for elem in pattern.pattern.elements:
                    if elem.node.variable:
                        variables.add(elem.node.variable)
                    if elem.relationship and elem.relationship.variable:
                        variables.add(elem.relationship.variable)
                continue
            if pattern.variable:
                variables.add(pattern.variable)
            for elem in pattern.elements:
                if elem.node.variable:
                    variables.add(elem.node.variable)
                if elem.relationship and elem.relationship.variable:
                    variables.add(elem.relationship.variable)
        return variables

    def _merge_property_filters(
        self,
        base: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Merge property filters, returning None when there is a conflict."""
        merged = dict(base)
        for key, value in extra.items():
            if key in merged and merged[key] != value:
                return None
            merged[key] = value
        return merged

    def _extract_property_filters(
        self,
        where_clause: WhereClause | None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Extract equality filters from WHERE for simple pushdown."""
        if where_clause is None:
            return {}
        filters = self._collect_property_filters(where_clause.condition, context=context)
        return filters

    def _collect_property_filters(
        self,
        expr: Expression,
        context: dict[str, Any] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Collect property = literal filters from AND-connected expressions."""
        if isinstance(expr, BinaryOp) and expr.operator == "AND":
            left = self._collect_property_filters(expr.left, context=context)
            right = self._collect_property_filters(expr.right, context=context)
            if not left:
                return right
            if not right:
                return left
            merged: dict[str, dict[str, Any]] = {}
            for var, props in left.items():
                merged[var] = dict(props)
            for var, props in right.items():
                if var not in merged:
                    merged[var] = dict(props)
                    continue
                merged_props = self._merge_property_filters(merged[var], props)
                if merged_props is None:
                    return {}
                merged[var] = merged_props
            return merged

        if isinstance(expr, BinaryOp) and expr.operator == "=":
            if isinstance(expr.left, PropertyAccess) and isinstance(expr.right, Literal):
                if expr.right.value is None:
                    return {}
                return {expr.left.variable: {expr.left.property: expr.right.value}}
            if isinstance(expr.right, PropertyAccess) and isinstance(expr.left, Literal):
                if expr.left.value is None:
                    return {}
                return {expr.right.variable: {expr.right.property: expr.left.value}}
            if context is not None:
                evaluator = self._make_evaluator(context)
                if isinstance(expr.left, PropertyAccess):
                    try:
                        value = evaluator.evaluate(expr.right)
                    except CypherExecutionError:
                        return {}
                    if value is None or isinstance(value, (Node, Relationship, list, dict)):
                        return {}
                    return {expr.left.variable: {expr.left.property: value}}
                if isinstance(expr.right, PropertyAccess):
                    try:
                        value = evaluator.evaluate(expr.left)
                    except CypherExecutionError:
                        return {}
                    if value is None or isinstance(value, (Node, Relationship, list, dict)):
                        return {}
                    return {expr.right.variable: {expr.right.property: value}}

        return {}

    def _match_pattern_function(self, pattern_func: PatternFunction) -> list[dict]:
        """Match shortestPath/allShortestPaths patterns."""
        inner = pattern_func.pattern
        if len(inner.elements) != 2:
            raise CypherExecutionError(
                f"{pattern_func.name} only supports a single relationship pattern"
            )

        source_elem = inner.elements[0]
        target_elem = inner.elements[1]
        rel_pattern = source_elem.relationship
        if rel_pattern is None:
            raise CypherExecutionError(
                f"{pattern_func.name} requires a relationship pattern"
            )

        source_properties = self._evaluate_properties(source_elem.node.properties, None)
        target_properties = self._evaluate_properties(target_elem.node.properties, None)
        rel_properties = self._evaluate_properties(rel_pattern.properties, None)

        start_nodes = self.db.match_nodes(
            labels=source_elem.node.labels if source_elem.node.labels else None,
            properties=source_properties if source_properties else None
        )
        end_nodes = self.db.match_nodes(
            labels=target_elem.node.labels if target_elem.node.labels else None,
            properties=target_properties if target_properties else None
        )

        all_paths = pattern_func.name == 'ALLSHORTESTPATHS'

        results = []
        for start_node in start_nodes:
            for end_node in end_nodes:
                paths = self._find_shortest_paths(
                    start_node,
                    end_node,
                    rel_pattern,
                    all_paths=all_paths,
                    rel_properties=rel_properties
                )
                for nodes_path, rels_path in paths:
                    bindings = {}
                    if source_elem.node.variable:
                        bindings[source_elem.node.variable] = start_node
                    if target_elem.node.variable:
                        bindings[target_elem.node.variable] = end_node
                    if rel_pattern.variable:
                        if rel_pattern.min_hops == 1 and rel_pattern.max_hops == 1:
                            bindings[rel_pattern.variable] = rels_path[0]
                        else:
                            bindings[rel_pattern.variable] = rels_path
                    if pattern_func.variable:
                        bindings[pattern_func.variable] = Path(
                            nodes=nodes_path,
                            relationships=rels_path
                        )
                    results.append(bindings)

        return results

    def _resolve_max_hops(self, rel_pattern: RelationshipPattern) -> int:
        """Resolve max hop limit for a relationship pattern."""
        max_hops = rel_pattern.max_hops
        if max_hops is None:
            max_hops = getattr(self.db, "cypher_max_hops", None)
            if not max_hops or max_hops <= 0:
                raise CypherExecutionError(
                    "Unbounded variable-length paths require a configured max hop limit"
                )
        return max_hops

    def _find_shortest_paths(
        self,
        start_node: Node,
        end_node: Node,
        rel_pattern: RelationshipPattern,
        all_paths: bool = False,
        rel_properties: dict[str, Any] | None = None
    ) -> list[tuple[list[Node], list[Relationship]]]:
        """Find shortest paths honoring relationship constraints."""
        min_hops = rel_pattern.min_hops
        max_hops = self._resolve_max_hops(rel_pattern)

        if start_node.id == end_node.id:
            if min_hops == 0:
                return [([start_node], [])]
            return []

        from collections import deque

        results: list[tuple[list[Node], list[Relationship]]] = []
        queue = deque([(start_node, [start_node], [], {start_node.id})])
        shortest_len = None

        while queue:
            current_node, nodes_path, rels_path, visited_ids = queue.popleft()
            depth = len(rels_path)

            if shortest_len is not None and depth >= shortest_len:
                continue
            if depth >= max_hops:
                continue

            for rel, next_node in self._get_next_relationships(
                current_node,
                rel_pattern,
                rel_properties=rel_properties
            ):
                if next_node.id in visited_ids:
                    continue
                new_nodes = nodes_path + [next_node]
                new_rels = rels_path + [rel]
                new_depth = depth + 1
                if new_depth > max_hops:
                    continue

                if next_node.id == end_node.id and new_depth >= min_hops:
                    if shortest_len is None or new_depth == shortest_len:
                        results.append((new_nodes, new_rels))
                        shortest_len = new_depth
                    elif new_depth < shortest_len:
                        results = [(new_nodes, new_rels)]
                        shortest_len = new_depth

                    if not all_paths:
                        return results[:1]

                if shortest_len is None or new_depth < shortest_len:
                    queue.append((next_node, new_nodes, new_rels, visited_ids | {next_node.id}))

        return results

    def _match_single_node(
        self,
        element: PatternElement,
        path_variable: str | None = None,
        property_filters: dict[str, dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Match a single node pattern.

        Args:
            element: PatternElement with node pattern only

        Returns:
            List of variable bindings
        """
        node_pattern = element.node
        if context and node_pattern.variable and node_pattern.variable in context:
            bound = self._resolve_bound_node(context.get(node_pattern.variable))
            if bound is None:
                return []
            evaluated_properties = self._evaluate_properties(node_pattern.properties, context)
            if not self._node_matches_pattern(bound, node_pattern, properties=evaluated_properties):
                return []
            result = {}
            result[node_pattern.variable] = bound
            if path_variable:
                result[path_variable] = Path(nodes=[bound], relationships=[])
            return [result]

        evaluated_properties = self._evaluate_properties(node_pattern.properties, context)
        merged_properties = evaluated_properties
        if property_filters and node_pattern.variable in property_filters:
            merged_properties = self._merge_property_filters(
                evaluated_properties,
                property_filters[node_pattern.variable]
            )
            if merged_properties is None:
                return []

        # Use match_nodes API
        nodes = self.db.match_nodes(
            labels=node_pattern.labels if node_pattern.labels else None,
            properties=merged_properties if merged_properties else None
        )

        # Build results
        results = []
        for node in nodes:
            result = {}
            if node_pattern.variable:
                result[node_pattern.variable] = node
            if path_variable:
                result[path_variable] = Path(nodes=[node], relationships=[])
            results.append(result)

        return results

    def _match_relationship_pattern(
        self,
        pattern: Pattern,
        path_variable: str | None = None,
        property_filters: dict[str, dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Match a relationship pattern like (a)-[r:TYPE]->(b).

        Args:
            pattern: Pattern with multiple elements connected by relationships

        Returns:
            List of variable bindings for (source, relationship, target) tuples
        """
        if len(pattern.elements) != 2:
            return self._match_multi_hop_relationship_pattern(
                pattern,
                path_variable=path_variable,
                context=context
            )

        source_elem = pattern.elements[0]
        target_elem = pattern.elements[1]
        rel_pattern = source_elem.relationship

        if rel_pattern is None:
            raise CypherExecutionError("Invalid relationship pattern")

        if rel_pattern.min_hops != 1 or rel_pattern.max_hops != 1:
            return self._match_variable_length_relationship_pattern(
                source_elem,
                target_elem,
                rel_pattern,
                path_variable=path_variable,
                context=context
            )

        filters = property_filters or {}
        source_properties = self._evaluate_properties(source_elem.node.properties, context)
        if source_elem.node.variable in filters:
            source_properties = self._merge_property_filters(
                source_properties,
                filters[source_elem.node.variable]
            )
            if source_properties is None:
                return []

        target_properties = self._evaluate_properties(target_elem.node.properties, context)
        if target_elem.node.variable in filters:
            target_properties = self._merge_property_filters(
                target_properties,
                filters[target_elem.node.variable]
            )
            if target_properties is None:
                return []

        rel_properties = self._evaluate_properties(rel_pattern.properties, context)
        if rel_pattern.variable and rel_pattern.variable in filters:
            rel_properties = self._merge_property_filters(
                rel_properties,
                filters[rel_pattern.variable]
            )
            if rel_properties is None:
                return []

        # Match source nodes
        source_nodes = None
        if context and source_elem.node.variable and source_elem.node.variable in context:
            bound_source = self._resolve_bound_node(context.get(source_elem.node.variable))
            if bound_source and self._node_matches_pattern(bound_source, source_elem.node, properties=source_properties):
                source_nodes = [bound_source]
            else:
                return []
        if source_nodes is None:
            source_nodes = self.db.match_nodes(
                labels=source_elem.node.labels if source_elem.node.labels else None,
                properties=source_properties if source_properties else None
            )

        target_ids = None
        if context and target_elem.node.variable and target_elem.node.variable in context:
            bound_target = self._resolve_bound_node(context.get(target_elem.node.variable))
            if bound_target and self._node_matches_pattern(bound_target, target_elem.node, properties=target_properties):
                target_ids = {bound_target.id}
            else:
                return []
        elif target_elem.node.labels or target_properties:
            target_nodes = self.db.match_nodes(
                labels=target_elem.node.labels if target_elem.node.labels else None,
                properties=target_properties if target_properties else None
            )
            target_ids = {node.id for node in target_nodes}

        # For each source node, find matching relationships
        results = []
        for source_node in source_nodes:
            # Get relationships based on direction
            if rel_pattern.direction == 'outgoing':
                rels = self.db.match_relationships(
                    source_id=source_node.id,
                    rel_type=rel_pattern.rel_type,
                    properties=rel_properties if rel_properties else None
                )
            elif rel_pattern.direction == 'incoming':
                rels = self.db.match_relationships(
                    target_id=source_node.id,
                    rel_type=rel_pattern.rel_type,
                    properties=rel_properties if rel_properties else None
                )
            else:  # 'both'
                outgoing = self.db.match_relationships(
                    source_id=source_node.id,
                    rel_type=rel_pattern.rel_type,
                    properties=rel_properties if rel_properties else None
                )
                incoming = self.db.match_relationships(
                    target_id=source_node.id,
                    rel_type=rel_pattern.rel_type,
                    properties=rel_properties if rel_properties else None
                )
                rels = outgoing + incoming

            # For each relationship, check target node matches
            for rel in rels:
                # Determine target node id based on direction
                if rel_pattern.direction == 'incoming':
                    target_id = rel.source_id
                elif rel_pattern.direction == 'both':
                    # Undirected: the "other" node is whichever end is not the source node
                    target_id = rel.target_id if rel.source_id == source_node.id else rel.source_id
                else:
                    target_id = rel.target_id

                if target_ids is not None and target_id not in target_ids:
                    continue

                target_node = self.db.get_node(target_id)
                if target_node is None:
                    continue

                # Check if target matches pattern
                if target_elem.node.labels:
                    if not all(label in target_node.labels for label in target_elem.node.labels):
                        continue

                if target_properties:
                    props_match = all(
                        target_node.properties.get(k) == v
                        for k, v in target_properties.items()
                    )
                    if not props_match:
                        continue

                # Build result
                result = {}
                if source_elem.node.variable:
                    result[source_elem.node.variable] = source_node
                if rel_pattern.variable:
                    result[rel_pattern.variable] = rel
                if target_elem.node.variable:
                    result[target_elem.node.variable] = target_node
                if path_variable:
                    result[path_variable] = Path(nodes=[source_node, target_node], relationships=[rel])

                results.append(result)

        return results

    def _match_multi_hop_relationship_pattern(
        self,
        pattern: Pattern,
        path_variable: str | None = None,
        context: dict[str, Any] | None = None
    ) -> list[dict]:
        """Match multi-hop relationship patterns across a chain."""
        elements = pattern.elements
        if len(elements) < 2:
            return []

        first_elem = elements[0]
        first_properties = self._evaluate_properties(first_elem.node.properties, context)
        start_nodes = self.db.match_nodes(
            labels=first_elem.node.labels if first_elem.node.labels else None,
            properties=first_properties if first_properties else None
        )

        results = []

        for start_node in start_nodes:
            base_bindings = {}
            if first_elem.node.variable:
                base_bindings[first_elem.node.variable] = start_node

            def walk(
                index: int,
                current_node: Node,
                bindings: dict,
                node_path: list[Node],
                rel_path: list[Relationship]
            ) -> None:
                if index == len(elements) - 1:
                    if path_variable:
                        bindings[path_variable] = Path(
                            nodes=node_path.copy(),
                            relationships=rel_path.copy()
                        )
                    results.append(bindings)
                    return

                rel_pattern = elements[index].relationship
                if rel_pattern is None:
                    raise CypherExecutionError("Missing relationship in multi-hop pattern")

                next_node_pattern = elements[index + 1].node

                rel_properties = self._evaluate_properties(rel_pattern.properties, context)
                next_properties = self._evaluate_properties(next_node_pattern.properties, context)

                for next_node, rel_value, segment_nodes in self._expand_relationship_segment(
                    current_node,
                    rel_pattern,
                    next_node_pattern,
                    rel_properties=rel_properties,
                    target_properties=next_properties
                ):
                    if next_node_pattern.variable and context and next_node_pattern.variable in context:
                        bound_next = self._resolve_bound_node(context.get(next_node_pattern.variable))
                        if bound_next is None or bound_next.id != next_node.id:
                            continue
                    new_bindings = bindings.copy()
                    if rel_pattern.variable:
                        new_bindings[rel_pattern.variable] = rel_value
                    if next_node_pattern.variable:
                        new_bindings[next_node_pattern.variable] = next_node
                    if isinstance(rel_value, list):
                        new_rel_path = rel_path + rel_value
                    else:
                        new_rel_path = rel_path + [rel_value]
                    new_node_path = node_path + segment_nodes[1:]
                    walk(index + 1, next_node, new_bindings, new_node_path, new_rel_path)

            walk(0, start_node, base_bindings, [start_node], [])

        return results

    def _expand_relationship_segment(
        self,
        start_node: Node,
        rel_pattern: RelationshipPattern,
        target_pattern: NodePattern,
        rel_properties: dict[str, Any] | None = None,
        target_properties: dict[str, Any] | None = None
    ) -> list[tuple[Node, Any, list[Node]]]:
        """Expand a relationship segment to matching end nodes."""
        if rel_pattern.min_hops == 1 and rel_pattern.max_hops == 1:
            results = []
            for rel, next_node in self._get_next_relationships(
                start_node,
                rel_pattern,
                rel_properties=rel_properties
            ):
                if not self._node_matches_pattern(
                    next_node,
                    target_pattern,
                    properties=target_properties
                ):
                    continue
                results.append((next_node, rel, [start_node, next_node]))
            return results

        return self._expand_variable_length_segment(
            start_node,
            rel_pattern,
            target_pattern,
            rel_properties=rel_properties,
            target_properties=target_properties
        )

    def _expand_variable_length_segment(
        self,
        start_node: Node,
        rel_pattern: RelationshipPattern,
        target_pattern: NodePattern,
        rel_properties: dict[str, Any] | None = None,
        target_properties: dict[str, Any] | None = None
    ) -> list[tuple[Node, list[Relationship], list[Node]]]:
        """Expand a variable-length segment with DFS and hop limits."""
        min_hops = rel_pattern.min_hops
        max_hops = rel_pattern.max_hops

        if max_hops is None:
            max_hops = getattr(self.db, "cypher_max_hops", None)
            if not max_hops or max_hops <= 0:
                raise CypherExecutionError(
                    "Unbounded variable-length paths require a configured max hop limit"
                )

        results = []

        def dfs(
            current_node: Node,
            depth: int,
            rel_path: list[Relationship],
            node_path: list[Node],
            visited_ids: set[int]
        ):
            if depth >= min_hops and self._node_matches_pattern(
                current_node,
                target_pattern,
                properties=target_properties
            ):
                results.append((current_node, rel_path.copy(), node_path.copy()))

            if depth == max_hops:
                return

            for rel, next_node in self._get_next_relationships(
                current_node,
                rel_pattern,
                rel_properties=rel_properties
            ):
                if next_node.id in visited_ids:
                    continue
                visited_ids.add(next_node.id)
                rel_path.append(rel)
                node_path.append(next_node)
                dfs(next_node, depth + 1, rel_path, node_path, visited_ids)
                node_path.pop()
                rel_path.pop()
                visited_ids.remove(next_node.id)

        dfs(start_node, 0, [], [start_node], {start_node.id})

        return results

    def _match_variable_length_relationship_pattern(
        self,
        source_elem: PatternElement,
        target_elem: PatternElement,
        rel_pattern: RelationshipPattern,
        path_variable: str | None = None,
        context: dict[str, Any] | None = None
    ) -> list[dict]:
        """Match variable-length relationship pattern (e.g., [:TYPE*1..3])."""
        min_hops = rel_pattern.min_hops
        max_hops = rel_pattern.max_hops

        if max_hops is None:
            max_hops = getattr(self.db, "cypher_max_hops", None)
            if not max_hops or max_hops <= 0:
                raise CypherExecutionError(
                    "Unbounded variable-length paths require a configured max hop limit"
                )

        source_properties = self._evaluate_properties(source_elem.node.properties, context)
        target_properties = self._evaluate_properties(target_elem.node.properties, context)
        rel_properties = self._evaluate_properties(rel_pattern.properties, context)

        # Match source nodes
        source_nodes = None
        if context and source_elem.node.variable and source_elem.node.variable in context:
            bound_source = self._resolve_bound_node(context.get(source_elem.node.variable))
            if bound_source and self._node_matches_pattern(bound_source, source_elem.node, properties=source_properties):
                source_nodes = [bound_source]
            else:
                return []
        if source_nodes is None:
            source_nodes = self.db.match_nodes(
                labels=source_elem.node.labels if source_elem.node.labels else None,
                properties=source_properties if source_properties else None
            )
        bound_target = None
        if context and target_elem.node.variable and target_elem.node.variable in context:
            bound_target = self._resolve_bound_node(context.get(target_elem.node.variable))

        results = []

        for source_node in source_nodes:
            def dfs(current_node, depth, rel_path, node_path, visited_ids):
                if depth >= min_hops:
                    if self._node_matches_pattern(
                        current_node,
                        target_elem.node,
                        properties=target_properties
                    ):
                        if not bound_target or current_node.id == bound_target.id:
                            result = {}
                            if source_elem.node.variable:
                                result[source_elem.node.variable] = source_node
                            if target_elem.node.variable:
                                result[target_elem.node.variable] = current_node
                            if rel_pattern.variable:
                                result[rel_pattern.variable] = rel_path.copy()
                            if path_variable:
                                result[path_variable] = Path(
                                    nodes=node_path.copy(),
                                    relationships=rel_path.copy()
                                )
                            results.append(result)

                if depth == max_hops:
                    return

                for rel, next_node in self._get_next_relationships(
                    current_node,
                    rel_pattern,
                    rel_properties=rel_properties
                ):
                    if next_node.id in visited_ids:
                        continue
                    visited_ids.add(next_node.id)
                    rel_path.append(rel)
                    node_path.append(next_node)
                    dfs(next_node, depth + 1, rel_path, node_path, visited_ids)
                    node_path.pop()
                    rel_path.pop()
                    visited_ids.remove(next_node.id)

            dfs(source_node, 0, [], [source_node], {source_node.id})

        return results

    def _node_matches_pattern(
        self,
        node: Node,
        pattern: NodePattern,
        properties: dict[str, Any] | None = None
    ) -> bool:
        """Check if a node matches a node pattern (labels and properties)."""
        if pattern.labels:
            if not all(label in node.labels for label in pattern.labels):
                return False

        props = properties if properties is not None else pattern.properties
        if props:
            for key, value in props.items():
                if node.properties.get(key) != value:
                    return False

        return True

    def _get_next_relationships(
        self,
        current_node: Node,
        rel_pattern: RelationshipPattern,
        rel_properties: dict[str, Any] | None = None
    ) -> list[tuple[Relationship, Node]]:
        """Get relationships and next nodes for variable-length traversal."""
        results = []

        if rel_pattern.direction in ('outgoing', 'both'):
            rels = self.db.match_relationships(
                source_id=current_node.id,
                rel_type=rel_pattern.rel_type
            )
            for rel in rels:
                if not self._relationship_matches_pattern(rel, rel_pattern, rel_properties):
                    continue
                next_node = self.db.get_node(rel.target_id)
                if next_node:
                    results.append((rel, next_node))

        if rel_pattern.direction in ('incoming', 'both'):
            rels = self.db.match_relationships(
                target_id=current_node.id,
                rel_type=rel_pattern.rel_type
            )
            for rel in rels:
                if not self._relationship_matches_pattern(rel, rel_pattern, rel_properties):
                    continue
                next_node = self.db.get_node(rel.source_id)
                if next_node:
                    results.append((rel, next_node))

        return results

    def _relationship_matches_pattern(
        self,
        rel: Relationship,
        rel_pattern: RelationshipPattern,
        properties: dict[str, Any] | None = None
    ) -> bool:
        """Check if a relationship matches type/properties constraints."""
        if rel_pattern.rel_type and rel.type != rel_pattern.rel_type:
            return False

        props = properties if properties is not None else rel_pattern.properties
        if props:
            for key, value in props.items():
                if rel.properties.get(key) != value:
                    return False

        return True

    def _filter_where(self, matches: list[dict], where_clause: WhereClause) -> list[dict]:
        """Filter matches using WHERE clause.

        Args:
            matches: List of variable bindings
            where_clause: WhereClause AST node

        Returns:
            Filtered list of matches
        """
        filtered = []

        for match in matches:
            evaluator = self._make_evaluator(match)
            try:
                result = evaluator.evaluate(where_clause.condition)
                if result:
                    filtered.append(match)
            except CypherExecutionError:
                # Skip matches that fail evaluation
                continue

        return filtered

    def _apply_return(self, matches: list[dict], return_clause: ReturnClause) -> list[dict]:
        """Apply RETURN projection to matches.

        Args:
            matches: List of variable bindings
            return_clause: ReturnClause AST node

        Returns:
            Projected results
        """
        # Check if any return item is an aggregation function
        has_aggregation = any(isinstance(item.expression, FunctionCall) for item in return_clause.items)

        if has_aggregation:
            # Aggregation mode: return a single row with aggregated values
            return self._apply_aggregation_return(matches, return_clause)
        else:
            # Normal mode: return one row per match
            return self._apply_normal_return(matches, return_clause)

    def _format_property_key(self, expr: PropertyAccess | PropertyLookup) -> str:
        """Format property access expressions for result keys."""
        if isinstance(expr, PropertyAccess):
            return f"{expr.variable}.{expr.property}"
        return self._format_property_lookup(expr)

    def _format_property_lookup(self, expr: PropertyLookup) -> str:
        base = expr.base_expr
        if isinstance(base, PropertyAccess):
            base_key = f"{base.variable}.{base.property}"
        elif isinstance(base, PropertyLookup):
            base_key = self._format_property_lookup(base)
        elif isinstance(base, Variable):
            base_key = base.name
        else:
            base_key = str(base)
        return f"{base_key}.{expr.property}"

    def _return_item_key(self, item: ReturnItem) -> str:
        """Compute the output key for a non-aggregated RETURN/WITH item."""
        if item.alias:
            return item.alias
        expr = item.expression
        if isinstance(expr, Variable):
            return expr.name
        if isinstance(expr, (PropertyAccess, PropertyLookup)):
            return self._format_property_key(expr)
        return str(expr)

    def _aggregation_result_key(self, item: ReturnItem) -> str:
        """Compute the output key for an aggregated RETURN item."""
        if item.alias:
            return item.alias

        func = item.expression
        func_name = func.function_name
        distinct_prefix = "DISTINCT " if func.distinct else ""
        if func.star:
            return f"{func_name}({distinct_prefix}*)"
        if len(func.arguments) == 1:
            argument = func.arguments[0]
            if isinstance(argument, Literal):
                return f"{func_name}({distinct_prefix}{argument.value})"
            if isinstance(argument, Variable):
                return f"{func_name}({distinct_prefix}{argument.name})"
            if isinstance(argument, PropertyAccess):
                return f"{func_name}({distinct_prefix}{argument.variable}.{argument.property})"
        return func_name

    def _apply_normal_return(self, matches: list[dict], return_clause: ReturnClause) -> list[dict]:
        """Apply non-aggregated RETURN projection."""
        results = []

        for match in matches:
            result = {}

            for item in return_clause.items:
                value = self._evaluate_return_expression(match, item.expression)
                result[self._return_item_key(item)] = value

            results.append(result)

        if return_clause.distinct:
            return self._apply_distinct(results)
        return results

    def _filter_rows(self, rows: list[dict], condition: Expression) -> list[dict]:
        """Keep rows whose condition evaluates truthy; drop rows that fail."""
        filtered = []
        for row in rows:
            evaluator = self._make_evaluator(row)
            try:
                if evaluator.evaluate(condition):
                    filtered.append(row)
            except CypherExecutionError:
                continue
        return filtered

    def _grouped_aggregation(
        self,
        items: list[ReturnItem],
        matches: list[dict],
        build_row,
    ) -> list[dict]:
        """Apply implicit GROUP BY: group matches by the non-aggregated items.

        Standard Cypher groups implicitly by the return/with items that are not
        aggregate functions. When there are no grouping items, the whole input
        collapses to a single global aggregate row.

        Args:
            items: RETURN/WITH items (mix of grouping keys and aggregates).
            matches: Input variable bindings to aggregate over.
            build_row: Callable that receives the matches belonging to one group
                and returns the projected output row for that group.

        Returns:
            One row per distinct grouping key (preserving first-seen order), or a
            single row when there are no grouping items.
        """
        grouping_items = [item for item in items if not isinstance(item.expression, FunctionCall)]

        if not grouping_items:
            return [build_row(matches)]

        groups: dict[Any, list[dict]] = {}
        order: list[Any] = []
        for match in matches:
            group_key = tuple(
                self._freeze_result(self._evaluate_return_expression(match, item.expression))
                for item in grouping_items
            )
            if group_key not in groups:
                groups[group_key] = []
                order.append(group_key)
            groups[group_key].append(match)

        return [build_row(groups[group_key]) for group_key in order]

    def _apply_aggregation_return(self, matches: list[dict], return_clause: ReturnClause) -> list[dict]:
        """Apply aggregated RETURN projection (COUNT, SUM, AVG, MIN, MAX).

        Implements implicit GROUP BY: non-aggregated return items become the
        grouping keys and aggregates are computed per group.
        """
        def build_row(group_matches: list[dict]) -> dict:
            result = {}
            for item in return_clause.items:
                if isinstance(item.expression, FunctionCall):
                    func = item.expression
                    value = self._calculate_aggregation(
                        func.function_name,
                        func.arguments,
                        group_matches,
                        func.distinct,
                        func.star,
                    )
                    result[self._aggregation_result_key(item)] = value
                else:
                    representative = group_matches[0] if group_matches else {}
                    result[self._return_item_key(item)] = self._evaluate_return_expression(
                        representative, item.expression
                    )
            return result

        results = self._grouped_aggregation(return_clause.items, matches, build_row)
        if return_clause.distinct:
            return self._apply_distinct(results)
        return results

    def _apply_distinct(self, rows: list[dict]) -> list[dict]:
        """Remove duplicate rows while preserving order."""
        seen = set()
        distinct_rows = []
        for row in rows:
            frozen = self._freeze_result(row)
            if frozen in seen:
                continue
            seen.add(frozen)
            distinct_rows.append(row)
        return distinct_rows

    def _evaluate_return_expression(self, match: dict, expr: Any) -> Any:
        """Evaluate a RETURN expression against a match row."""
        if isinstance(expr, PatternComprehension):
            return self._evaluate_pattern_comprehension(expr, match)

        if isinstance(expr, Variable):
            var_name = expr.name
            if var_name in match:
                return self._serialize_value(match[var_name])
            return None

        if isinstance(expr, PropertyAccess):
            var_name = expr.variable
            prop_name = expr.property
            if var_name not in match:
                return None
            obj = match[var_name]
            if hasattr(obj, 'properties'):
                return obj.properties.get(prop_name)
            if isinstance(obj, dict):
                return obj.get(prop_name)
            return None

        evaluator = self._make_evaluator(match)
        value = evaluator.evaluate(expr)
        return self._serialize_value(value)

    def _serialize_value(self, value: Any) -> Any:
        """Convert model instances to dictionaries for output."""
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return value

    def _calculate_aggregation(
        self,
        func_name: str,
        arguments: list[Any],
        matches: list[dict],
        distinct: bool = False,
        star: bool = False,
    ) -> Any:
        """Calculate aggregation function value.

        Args:
            func_name: Function name (COUNT, SUM, AVG, MIN, MAX)
            argument: Expression to aggregate over (or None for COUNT(*))
            matches: List of variable bindings

        Returns:
            Aggregated value
        """
        if func_name != 'COUNT' and star:
            raise CypherExecutionError(f"{func_name}(*) is not supported")

        if func_name == 'COUNT':
            if star:
                # COUNT(*) - count all rows
                return len(matches)
            else:
                if not arguments:
                    raise CypherExecutionError("COUNT() expects an argument or *")
                # COUNT(expr) - count non-null values
                values = []
                for match in matches:
                    evaluator = self._make_evaluator(match)
                    try:
                        value = evaluator.evaluate(arguments[0]) if arguments else None
                        if value is not None:
                            values.append(value)
                    except:
                        pass
                if distinct:
                    values = self._distinct_values(values)
                return len(values)

        elif func_name in ('SUM', 'AVG', 'MIN', 'MAX', 'COLLECT', 'STDDEV', 'PERCENTILECONT'):
            if not arguments:
                raise CypherExecutionError(f"{func_name}() expects at least 1 argument")
            # Collect values
            values = []
            for match in matches:
                evaluator = self._make_evaluator(match)
                try:
                    value = evaluator.evaluate(arguments[0]) if arguments else None
                    if func_name == 'COLLECT':
                        values.append(value)
                    elif value is not None:
                        values.append(value)
                except:
                    pass

            if distinct:
                values = self._distinct_values(values)

            if func_name == 'COLLECT':
                return values

            if not values:
                return None

            if func_name == 'SUM':
                return sum(values)
            elif func_name == 'AVG':
                return sum(values) / len(values)
            elif func_name == 'MIN':
                return min(values)
            elif func_name == 'MAX':
                return max(values)
            elif func_name == 'STDDEV':
                if len(values) == 1:
                    return 0.0
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                return variance ** 0.5
            elif func_name == 'PERCENTILECONT':
                if len(arguments) < 2:
                    raise CypherExecutionError("PERCENTILECONT requires value and percentile")
                percentile = evaluator.evaluate(arguments[1])
                if percentile is None:
                    return None
                if not isinstance(percentile, (int, float)):
                    raise CypherExecutionError("PERCENTILECONT percentile must be a number")
                if percentile < 0 or percentile > 1:
                    raise CypherExecutionError("PERCENTILECONT percentile must be between 0 and 1")
                sorted_values = sorted(values)
                if len(sorted_values) == 1:
                    return float(sorted_values[0])
                index = percentile * (len(sorted_values) - 1)
                lower_index = int(index)
                upper_index = min(lower_index + 1, len(sorted_values) - 1)
                lower = sorted_values[lower_index]
                upper = sorted_values[upper_index]
                if lower_index == upper_index:
                    return float(lower)
                fraction = index - lower_index
                return lower + (upper - lower) * fraction

        raise CypherExecutionError(f"Unknown aggregation function: {func_name}")

    def _distinct_values(self, values: list[Any]) -> list[Any]:
        """Return distinct values preserving first occurrence order."""
        seen = set()
        distinct_values = []
        for value in values:
            frozen = self._freeze_result(value)
            if frozen in seen:
                continue
            seen.add(frozen)
            distinct_values.append(value)
        return distinct_values

    def _execute_delete(self, matches: list[dict], delete_clause) -> list[dict]:
        """Execute DELETE clause to delete nodes and relationships.

        Args:
            matches: List of variable bindings
            delete_clause: DeleteClause AST node

        Returns:
            List with count of deleted items
        """
        deleted_nodes = 0
        deleted_rels = 0

        for match in matches:
            for var_name in delete_clause.variables:
                if var_name not in match:
                    raise CypherExecutionError(f"Variable '{var_name}' not found for DELETE")

                obj = match[var_name]

                # Check if it's a Node or Relationship
                if hasattr(obj, 'source_id'):  # It's a Relationship
                    self.db.delete_relationship(obj.id)
                    deleted_rels += 1
                elif hasattr(obj, 'labels'):  # It's a Node
                    self.db.delete_node(obj.id)
                    deleted_nodes += 1
                else:
                    raise CypherExecutionError(
                        f"Variable '{var_name}' is not a node or relationship"
                    )

        return [{'deleted_nodes': deleted_nodes, 'deleted_relationships': deleted_rels}]

    def _execute_set(self, matches: list[dict], set_clause) -> None:
        """Execute SET clause to update node properties.

        Args:
            matches: List of variable bindings
            set_clause: SetClause AST node
        """
        for match in matches:
            for set_item in set_clause.items:
                var_name = set_item.variable
                prop_name = set_item.property

                if var_name not in match:
                    raise CypherExecutionError(f"Variable '{var_name}' not found for SET")

                obj = match[var_name]

                # Evaluate the value expression
                evaluator = self._make_evaluator(match)
                value = evaluator.evaluate(set_item.value)

                if prop_name is None:
                    if set_item.operator == "+=":
                        if not isinstance(value, dict):
                            raise CypherExecutionError("SET += expects a map value")
                        merged = obj.properties.copy()
                        merged.update(value)
                        self._apply_properties_update(obj, merged, replace=True)
                    elif set_item.operator == "=":
                        if not isinstance(value, dict):
                            raise CypherExecutionError("SET = expects a map value")
                        self._apply_properties_update(obj, value, replace=True)
                    else:
                        raise CypherExecutionError(f"Unsupported SET operator: {set_item.operator}")
                else:
                    new_properties = obj.properties.copy()
                    new_properties[prop_name] = value
                    self._apply_properties_update(obj, new_properties, replace=True)

    def _apply_properties_update(self, obj: Any, properties: dict, replace: bool) -> None:
        """Apply property updates to a node or relationship."""
        if not hasattr(obj, 'properties'):
            raise CypherExecutionError("SET target does not have properties")
        if hasattr(obj, 'labels'):
            if replace:
                self.db.replace_node_properties(obj.id, properties)
            else:
                self.db.update_node_properties(obj.id, properties)
            obj.properties = properties
            return
        if hasattr(obj, 'source_id'):
            if replace:
                self.db.replace_relationship_properties(obj.id, properties)
            else:
                self.db.update_relationship_properties(obj.id, properties)
            obj.properties = properties
            return
        raise CypherExecutionError("SET target is not a node or relationship")

    def _execute_load_csv(self, clause: LoadCsvClause, input_results: list[dict]) -> list[dict]:
        """Execute LOAD CSV clause to produce rows."""
        if not input_results:
            input_results = [{}]

        output = []
        for row in input_results:
            evaluator = self._make_evaluator(row)
            source = evaluator.evaluate(clause.source)
            if not isinstance(source, str):
                raise CypherExecutionError("LOAD CSV source must be a string")

            rows = self._read_csv_rows(source, clause.with_headers)
            for csv_row in rows:
                merged = row.copy()
                merged[clause.variable] = csv_row
                output.append(merged)
        return output

    def _read_csv_rows(self, source: str, with_headers: bool) -> list[Any]:
        """Read CSV rows from a URL or local path."""
        if source.startswith("http://") or source.startswith("https://"):
            with urllib.request.urlopen(source) as response:
                data = response.read().decode("utf-8")
            handle = io.StringIO(data)
            return self._parse_csv(handle, with_headers)

        if source.startswith("file://"):
            source = source[len("file://"):]
        if not os.path.exists(source):
            raise CypherExecutionError(f"LOAD CSV file not found: {source}")
        with open(source, "r", encoding="utf-8") as handle:
            data = handle.read()
        return self._parse_csv(io.StringIO(data), with_headers)

    def _parse_csv(self, handle, with_headers: bool) -> list[Any]:
        if with_headers:
            reader = csv.DictReader(handle)
            cleaned_rows = []
            for row in reader:
                cleaned = {key: value for key, value in row.items() if isinstance(key, str)}
                cleaned_rows.append(cleaned)
            return cleaned_rows
        reader = csv.reader(handle)
        return [list(row) for row in reader]

    def _execute_remove(self, matches: list[dict], remove_clause) -> None:
        """Execute REMOVE clause to remove properties or labels from nodes.

        Args:
            matches: List of variable bindings
            remove_clause: RemoveClause AST node

        Supports:
            REMOVE n.property  - Remove a property
            REMOVE n:Label     - Remove a label
        """
        for match in matches:
            for remove_item in remove_clause.items:
                var_name = remove_item.variable

                if var_name not in match:
                    raise CypherExecutionError(f"Variable '{var_name}' not found for REMOVE")

                obj = match[var_name]

                if not hasattr(obj, 'labels'):  # Must be a Node
                    raise CypherExecutionError(
                        f"REMOVE is only supported for nodes, not relationships"
                    )

                if remove_item.property:
                    # Remove property: REMOVE n.age
                    prop_name = remove_item.property

                    # Check if property exists
                    if prop_name not in obj.properties:
                        # Property doesn't exist - silently continue (Neo4j behavior)
                        continue

                    # Remove from properties dict
                    new_properties = obj.properties.copy()
                    del new_properties[prop_name]

                    # Update in database (direct SQL since update_node_properties does merge)
                    properties_json = orjson.dumps(new_properties).decode('utf-8')
                    self.db.conn.execute(
                        "UPDATE nodes SET properties = ? WHERE id = ?",
                        (properties_json, obj.id)
                    )
                    if not self.db._in_transaction:
                        self.db.conn.commit()

                    # Update object in memory
                    obj.properties = new_properties

                elif remove_item.label:
                    # Remove label: REMOVE n:OldLabel
                    label_name = remove_item.label

                    # Check if label exists
                    if label_name not in obj.labels:
                        # Label doesn't exist - silently continue (Neo4j behavior)
                        continue

                    # Remove label using database API
                    self.db.remove_labels(obj.id, [label_name])
                    # Update object in memory
                    obj.labels = [l for l in obj.labels if l != label_name]

                else:
                    raise CypherExecutionError(
                        f"REMOVE item must have either property or label"
                    )

    def _apply_order_by(self, matches: list[dict], order_by_clause) -> list[dict]:
        """Apply ORDER BY clause to sort results.

        Args:
            matches: List of result dictionaries
            order_by_clause: OrderByClause AST node

        Returns:
            Sorted list of results
        """
        def get_sort_key(match):
            """Generate sort key tuple for a match."""
            keys = []
            for item in order_by_clause.items:
                # Evaluate the expression for this match
                evaluator = self._make_evaluator(match)
                try:
                    value = evaluator.evaluate(item.expression)
                    # Handle None values (sort to end)
                    if value is None:
                        # Use a large value for None so it sorts to the end
                        value = (float('inf'),)
                    else:
                        # For DESC, negate numeric values or use reverse wrapper
                        if not item.ascending:
                            # Wrap value so it sorts in reverse
                            if isinstance(value, (int, float)):
                                value = (-value,)
                            else:
                                # For strings, we can't easily negate, so use a wrapper
                                value = (ReverseWrapper(value),)
                        else:
                            value = (value,)
                    keys.append(value)
                except:
                    # If evaluation fails, treat as None (end of list)
                    keys.append((float('inf'),))
            return tuple(keys)

        # Sort without reverse - direction is handled in the key
        sorted_matches = sorted(matches, key=get_sort_key)

        return sorted_matches

    def _execute_with(self, clause: WithClause, input_results: list[dict]) -> list[dict]:
        """Execute WITH clause as a pipeline stage.

        Args:
            clause: WithClause AST node
            input_results: Results from previous stage

        Returns:
            Transformed/filtered results

        The WITH clause:
        1. Projects specific items from input (like RETURN)
        2. Can apply WHERE filter
        3. Can apply ORDER BY, SKIP, LIMIT
        4. Results pass to next stage
        """
        if not input_results:
            return []

        has_aggregation = any(
            isinstance(item.expression, FunctionCall)
            for item in clause.items
        )

        # Without aggregation, WHERE filters the input rows (so it can reference
        # any bound variable). With aggregation it behaves like HAVING and runs
        # after grouping (so it can reference aggregate aliases) — see below.
        if clause.where_clause and not has_aggregation:
            input_results = self._filter_rows(input_results, clause.where_clause.condition)

        if has_aggregation:
            def build_with_row(group_rows: list[dict]) -> dict:
                representative = group_rows[0] if group_rows else {}
                result = {}
                for item in clause.items:
                    if isinstance(item.expression, FunctionCall):
                        value = self._calculate_aggregation(
                            item.expression.function_name,
                            item.expression.arguments,
                            group_rows,
                            item.expression.distinct,
                            item.expression.star
                        )
                    elif isinstance(item.expression, Variable):
                        # Preserve the bound object (e.g. Node) for downstream clauses.
                        value = representative.get(item.expression.name)
                    else:
                        evaluator = self._make_evaluator(representative)
                        value = evaluator.evaluate(item.expression)

                    if item.alias:
                        result[item.alias] = value
                    elif isinstance(item.expression, FunctionCall):
                        distinct_prefix = "DISTINCT " if item.expression.distinct else ""
                        if item.expression.star:
                            key = f"{item.expression.function_name}({distinct_prefix}*)"
                        elif item.expression.arguments and isinstance(item.expression.arguments[0], PropertyAccess):
                            arg = item.expression.arguments[0]
                            key = f"{item.expression.function_name}({distinct_prefix}{arg.variable}.{arg.property})"
                        else:
                            key = f"{item.expression.function_name}({distinct_prefix}...)"
                        result[key] = value
                    else:
                        result[self._return_item_key(item)] = value
                return result

            projected_results = self._grouped_aggregation(clause.items, input_results, build_with_row)
            # HAVING: filter the grouped rows by the post-aggregation WHERE.
            if clause.where_clause:
                projected_results = self._filter_rows(projected_results, clause.where_clause.condition)
        else:
            projected_results = []
            for match in input_results:
                result = {}
                for item in clause.items:
                    evaluator = self._make_evaluator(match)

                    if isinstance(item.expression, (PropertyAccess, PropertyLookup)):
                        value = evaluator.evaluate(item.expression)
                    elif isinstance(item.expression, Variable):
                        var_name = item.expression.name
                        value = match.get(var_name)
                    else:
                        value = evaluator.evaluate(item.expression)

                    if item.alias:
                        result[item.alias] = value
                    elif isinstance(item.expression, (PropertyAccess, PropertyLookup)):
                        key = self._format_property_key(item.expression)
                        result[key] = value
                    elif isinstance(item.expression, Variable):
                        result[item.expression.name] = value
                    else:
                        result[str(item.expression)] = value

                projected_results.append(result)

        if clause.distinct:
            projected_results = self._apply_distinct(projected_results)

        if clause.order_by_clause:
            projected_results = self._apply_order_by(projected_results, clause.order_by_clause)

        if clause.skip_clause:
            projected_results = projected_results[clause.skip_clause.count:]

        if clause.limit_clause:
            projected_results = projected_results[:clause.limit_clause.count]

        if clause.return_clause:
            projected_results = self._apply_normal_return(projected_results, clause.return_clause)

        return projected_results

    def _execute_create_index(self, clause: CreateIndexClause) -> list[dict]:
        """Execute CREATE INDEX clause."""
        if clause.entity == "node":
            name = self.db.create_node_index(clause.label_or_type, clause.property, unique=clause.unique)
        else:
            name = self.db.create_relationship_index(clause.label_or_type, clause.property, unique=clause.unique)
        return [{"name": name}]

    def _execute_drop_index(self, clause: DropIndexClause) -> list[dict]:
        """Execute DROP INDEX clause."""
        self.db.drop_index(clause.name)
        return []

    def _execute_show_indexes(self, clause: ShowIndexesClause) -> list[dict]:
        """Execute SHOW INDEXES clause."""
        indexes = self.db.list_indexes()
        if clause.where_expr is None:
            return indexes
        filtered = []
        for index in indexes:
            evaluator = self._make_evaluator(index)
            try:
                if evaluator.evaluate(clause.where_expr):
                    filtered.append(index)
            except CypherExecutionError:
                continue
        return filtered

    def _execute_create_constraint(self, clause: CreateConstraintClause) -> list[dict]:
        """Execute CREATE CONSTRAINT clause."""
        if clause.constraint_type == "UNIQUE":
            if clause.entity == "node":
                name = self.db.create_node_uniqueness_constraint(
                    clause.label_or_type, clause.property, clause.name, clause.if_not_exists
                )
            else:
                name = self.db.create_relationship_uniqueness_constraint(
                    clause.label_or_type, clause.property, clause.name, clause.if_not_exists
                )
        elif clause.constraint_type == "EXISTS":
            if clause.entity == "node":
                name = self.db.create_node_existence_constraint(
                    clause.label_or_type, clause.property, clause.name, clause.if_not_exists
                )
            else:
                name = self.db.create_relationship_existence_constraint(
                    clause.label_or_type, clause.property, clause.name, clause.if_not_exists
                )
        else:
            if clause.entity == "node":
                name = self.db.create_node_type_constraint(
                    clause.label_or_type, clause.property, clause.type_name or "", clause.name, clause.if_not_exists
                )
            else:
                name = self.db.create_relationship_type_constraint(
                    clause.label_or_type, clause.property, clause.type_name or "", clause.name, clause.if_not_exists
                )
        return [{"name": name}]

    def _execute_drop_constraint(self, clause: DropConstraintClause) -> list[dict]:
        """Execute DROP CONSTRAINT clause."""
        self.db.drop_constraint(clause.name, clause.if_exists)
        return []

    def _execute_show_constraints(self, clause: ShowConstraintsClause) -> list[dict]:
        """Execute SHOW CONSTRAINTS clause."""
        constraints = self.db.list_constraints()
        if clause.where_expr is None:
            return constraints
        filtered = []
        for constraint in constraints:
            evaluator = self._make_evaluator(constraint)
            try:
                if evaluator.evaluate(clause.where_expr):
                    filtered.append(constraint)
            except CypherExecutionError:
                continue
        return filtered

    def _execute_foreach(self, clause: ForeachClause, input_results: list[dict]) -> list[dict]:
        """Execute FOREACH clause for updates."""
        if not input_results:
            input_results = [{}]

        for row in input_results:
            evaluator = self._make_evaluator(row)
            values = evaluator.evaluate(clause.list_expr)
            if values is None:
                values = []
            if not isinstance(values, (list, tuple)):
                raise CypherExecutionError("FOREACH list expression must be a list")
            for item in values:
                local_row = row.copy()
                local_row[clause.variable] = item
                for action in clause.actions:
                    if isinstance(action, CreateClause):
                        created = self._execute_create(action, context=[local_row])
                        if created:
                            local_row.update(created[0])
                    elif isinstance(action, MergeClause):
                        merged = self._execute_merge(action, context=[local_row])
                        if merged:
                            local_row.update(merged[0])
                    elif isinstance(action, SetClause):
                        self._execute_set([local_row], action)
                    elif isinstance(action, RemoveClause):
                        self._execute_remove([local_row], action)
                    elif isinstance(action, DeleteClause):
                        self._execute_delete([local_row], action)
                    else:
                        raise CypherExecutionError(
                            f"Unsupported FOREACH action: {type(action)}"
                        )

        return input_results

    # =========================================================================
    # Dump / Restore
    # =========================================================================

    def dump(self, file_path: str) -> None:
        """Dump the entire database to a Cypher script file.

        The generated script uses a temporary `_dump_id` property on nodes
        to link relationships during restore. This property is removed at
        the end of the script.

        Args:
            file_path: Path to the output .cypher file.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("// Grafito Database Dump\n")
            f.write("// Generated automatically - do not edit manually\n\n")

            # Constraints
            constraints = self.db.list_constraints()
            if constraints:
                f.write("// Constraints\n")
                for c in constraints:
                    entity = c["entity"]
                    label_or_type = c["label_or_type"]
                    prop = c["property"]
                    ctype = c["type"]
                    type_name = c.get("type_name")

                    if entity == "node":
                        pattern = f"(n:{label_or_type})"
                        require_expr = f"n.{prop}"
                    else:
                        pattern = f"()-[r:{label_or_type}]-()"
                        require_expr = f"r.{prop}"

                    if ctype == "UNIQUE":
                        f.write(f"CREATE CONSTRAINT FOR {pattern} REQUIRE {require_expr} IS UNIQUE;\n")
                    elif ctype == "EXISTS":
                        f.write(f"CREATE CONSTRAINT FOR {pattern} REQUIRE {require_expr} IS NOT NULL;\n")
                    elif ctype == "TYPE" and type_name:
                        f.write(f"CREATE CONSTRAINT FOR {pattern} REQUIRE {require_expr} IS {type_name};\n")
                f.write("\n")

            # Indexes
            indexes = self.db.list_indexes()
            if indexes:
                f.write("// Indexes\n")
                for idx in indexes:
                    entity = idx["entity"]
                    label_or_type = idx["label_or_type"] or "Node"
                    prop = idx["property"]
                    unique = idx.get("unique", False)

                    unique_kw = "UNIQUE " if unique else ""
                    if entity == "node":
                        f.write(f"CREATE {unique_kw}INDEX FOR (n:{label_or_type}) ON (n.{prop});\n")
                    else:
                        f.write(f"CREATE {unique_kw}INDEX FOR ()-[r:{label_or_type}]-() ON (r.{prop});\n")
                f.write("\n")

            # Nodes
            all_nodes = self.db.match_nodes()
            if all_nodes:
                f.write("// Nodes\n")
                for node in all_nodes:
                    labels_str = ":".join(node.labels) if node.labels else ""
                    props = dict(node.properties)
                    props["_dump_id"] = node.id
                    props_str = self._format_properties(props)
                    if labels_str:
                        f.write(f"CREATE (:{labels_str} {props_str});\n")
                    else:
                        f.write(f"CREATE ({props_str});\n")
                f.write("\n")

            # Relationships
            all_rels = self.db.match_relationships()
            if all_rels:
                f.write("// Relationships\n")
                for rel in all_rels:
                    props_str = self._format_properties(rel.properties) if rel.properties else ""
                    rel_part = f"[:{rel.type}]" if not props_str else f"[:{rel.type} {props_str}]"
                    f.write(
                        f"MATCH (a), (b) WHERE a._dump_id = {rel.source_id} AND b._dump_id = {rel.target_id} "
                        f"CREATE (a)-{rel_part}->(b);\n"
                    )
                f.write("\n")

            # Cleanup _dump_id
            if all_nodes:
                f.write("// Cleanup\n")
                f.write("MATCH (n) REMOVE n._dump_id;\n")

    def restore(self, file_path: str, clear_existing: bool = True) -> None:
        """Restore the database from a Cypher script file.

        The script is fully parsed before any data is modified. If parsing
        fails, the database remains unchanged.

        Args:
            file_path: Path to the .cypher file.
            clear_existing: If True, delete all existing data before restore.

        Raises:
            CypherSyntaxError: If any statement in the script is invalid.
        """
        from .lexer import Lexer
        from .parser import Parser

        with open(file_path, "r", encoding="utf-8") as f:
            script = f.read()

        # Split statements
        statements = self.db._split_cypher_statements(script)

        # Parse all statements first (validation)
        parsed = []
        for stmt in statements:
            # Remove comment lines from within the statement
            lines = [line for line in stmt.split('\n') if not line.strip().startswith('//')]
            stmt = '\n'.join(lines).strip()
            if not stmt:
                continue
            lexer = Lexer(stmt)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            parsed.append(ast)

        # Clear existing data if requested
        if clear_existing:
            for rel in self.db.match_relationships():
                self.db.delete_relationship(rel.id)
            for node in self.db.match_nodes():
                self.db.delete_node(node.id)
            for constraint in self.db.list_constraints():
                self.db.drop_constraint(constraint["name"], if_exists=True)
            for idx in self.db.list_indexes():
                self.db.drop_index(idx["name"])

        # Execute parsed statements
        for ast in parsed:
            self.execute(ast)

    def _format_properties(self, props: dict) -> str:
        """Format a properties dict as a Cypher map literal."""
        if not props:
            return "{}"
        parts = []
        for k, v in props.items():
            parts.append(f"{k}: {self._format_value(v)}")
        return "{" + ", ".join(parts) + "}"

    def _format_value(self, value) -> str:
        """Format a Python value as a Cypher literal."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace("'", "\\'")
            return f"'{escaped}'"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, list):
            items = ", ".join(self._format_value(v) for v in value)
            return f"[{items}]"
        if isinstance(value, dict):
            parts = [f"{k}: {self._format_value(v)}" for k, v in value.items()]
            return "{" + ", ".join(parts) + "}"
        return repr(value)


class ReverseWrapper:
    """Wrapper to reverse sort order for non-numeric values."""
    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        if isinstance(other, ReverseWrapper):
            return self.value > other.value
        return False

    def __le__(self, other):
        if isinstance(other, ReverseWrapper):
            return self.value >= other.value
        return False

    def __gt__(self, other):
        if isinstance(other, ReverseWrapper):
            return self.value < other.value
        return False

    def __ge__(self, other):
        if isinstance(other, ReverseWrapper):
            return self.value <= other.value
        return False

    def __eq__(self, other):
        if isinstance(other, ReverseWrapper):
            return self.value == other.value
        return False
