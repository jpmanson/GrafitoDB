import json
import pytest

from grafito import GrafitoDatabase


def test_call_vector_search_procedure():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2, options={"store_embeddings": True})
    a = db.create_node(labels=["Person"], properties={"name": "Alice"})
    b = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(a.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(b.id, [0.0, 1.0], index="people_vec")

    results = db.execute("""
        CALL db.vector.search('people_vec', [1.0, 0.0], 1) YIELD node, score
        RETURN node, score
    """)
    assert len(results) == 1
    assert results[0]["node"]["id"] == a.id
    db.close()


def test_call_vector_search_with_label_filter():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2, options={"store_embeddings": True})
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    acme = db.create_node(labels=["Company"], properties={"name": "Acme"})
    db.upsert_embedding(alice.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(acme.id, [1.0, 0.0], index="people_vec")

    results = db.execute("""
        CALL db.vector.search('people_vec', [1.0, 0.0], 5, {labels: ['Person']}) YIELD node
        RETURN node.name AS name
    """)
    assert results == [{"name": "Alice"}]
    db.close()


def test_call_vector_search_with_property_filter():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2, options={"store_embeddings": True})
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(alice.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(bob.id, [1.0, 0.0], index="people_vec")

    results = db.execute("""
        CALL db.vector.search('people_vec', [1.0, 0.0], 5, {properties: {name: 'Alice'}}) YIELD node
        RETURN node.name AS name
    """)
    assert results == [{"name": "Alice"}]
    db.close()


def test_call_vector_search_with_rerank_and_candidate_multiplier():
    pytest.importorskip("faiss")
    db = GrafitoDatabase(':memory:')
    db.create_vector_index(
        "people_vec",
        dim=2,
        backend="faiss",
        options={"store_embeddings": True, "metric": "l2"},
    )
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    acme = db.create_node(labels=["Company"], properties={"name": "Acme"})
    db.upsert_embedding(bob.id, [2.0, 0.0], index="people_vec")
    db.upsert_embedding(alice.id, [3.0, 0.0], index="people_vec")
    db.upsert_embedding(acme.id, [1.0, 0.0], index="people_vec")

    results = db.execute("""
        CALL db.vector.search(
            'people_vec',
            [1.0, 0.0],
            1,
            {labels: ['Person'], candidate_multiplier: 1}
        ) YIELD node
        RETURN node.name AS name
    """)
    assert results == []

    results = db.execute("""
        CALL db.vector.search(
            'people_vec',
            [1.0, 0.0],
            1,
            {labels: ['Person'], candidate_multiplier: 2, rerank: true}
        ) YIELD node
        RETURN node.name AS name
    """)
    assert results == [{"name": "Bob"}]
    db.close()


def test_call_vector_search_with_custom_reranker():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2, options={"store_embeddings": True})
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(alice.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(bob.id, [1.0, 0.0], index="people_vec")

    def reverse_reranker(query_vector, candidates):
        return [{"id": item["id"], "score": 1.0} for item in reversed(candidates)]

    db.register_reranker("reverse", reverse_reranker)
    results = db.execute("""
        CALL db.vector.search('people_vec', [1.0, 0.0], 2, {reranker: 'reverse'})
        YIELD node
        RETURN node.name AS name
    """)
    assert results == [{"name": "Bob"}, {"name": "Alice"}]
    db.close()


def test_call_vector_search_unknown_reranker_raises():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2)
    with pytest.raises(Exception, match="Unknown reranker"):
        db.execute("""
            CALL db.vector.search('people_vec', [1.0, 0.0], 1, {reranker: 'missing'})
            YIELD node
            RETURN node
        """)
    db.close()


def test_call_vector_search_with_property_operators():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2, options={"store_embeddings": True})
    alice = db.create_node(labels=["Person"], properties={"name": "Alice", "age": 30})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob", "age": 42})
    db.upsert_embedding(alice.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(bob.id, [1.0, 0.0], index="people_vec")

    results = db.execute("""
        CALL db.vector.search(
            'people_vec',
            [1.0, 0.0],
            5,
            {properties: {age: {gt: 35}}}
        ) YIELD node
        RETURN node.name AS name
    """)
    assert results == [{"name": "Bob"}]
    db.close()


def test_call_vector_search_with_property_or_filter():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2, options={"store_embeddings": True})
    alice = db.create_node(labels=["Person"], properties={"name": "Alice", "age": 30})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob", "age": 42})
    db.upsert_embedding(alice.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(bob.id, [1.0, 0.0], index="people_vec")

    results = db.execute("""
        CALL db.vector.search(
            'people_vec',
            [1.0, 0.0],
            5,
            {properties: {or: [{age: {lt: 35}}, {name: {contains: 'Bo'}}]}}
        ) YIELD node
        RETURN node.name AS name
    """)
    assert results == [{"name": "Alice"}, {"name": "Bob"}]
    db.close()


def test_call_vector_search_default_k_from_index():
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2, options={"default_k": 1})
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(alice.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(bob.id, [0.0, 1.0], index="people_vec")

    results = db.execute("""
        CALL db.vector.search('people_vec', [1.0, 0.0]) YIELD node
        RETURN node.name AS name
    """)
    assert results == [{"name": "Alice"}]
    db.close()


def test_apoc_load_html_with_list_index_and_property_lookup(tmp_path):
    html = """
    <table class="wikitable">
        <tbody>
            <tr>
                <td>Brand A</td><td>Type A</td><td>5%</td><td>Brewery A</td><td>1900</td>
            </tr>
            <tr>
                <td>Brand B</td><td>Type B</td><td>6%</td><td>Brewery B</td><td>1950</td>
            </tr>
        </tbody>
    </table>
    """
    path = tmp_path / "beers.html"
    path.write_text(html, encoding="utf-8")

    db = GrafitoDatabase(":memory:")
    results = db.execute(f"""
        WITH '{path.as_uri()}' AS url
        CALL apoc.load.html(url, {{
            brand: "table.wikitable tbody tr td:eq(0)",
            beertype: "table.wikitable tbody tr td:eq(1)"
        }}) YIELD value
        WITH value, size(value.brand) AS rangeup
        UNWIND range(0, rangeup - 1) AS i
        RETURN value.brand[i].text AS brand, value.beertype[i].text AS beertype
    """)
    assert results == [
        {"brand": "Brand A", "beertype": "Type A"},
        {"brand": "Brand B", "beertype": "Type B"},
    ]
    db.close()


def test_apoc_load_xml_with_xpath(tmp_path):
    xml = """
    <root>
        <beer><name>Brand A</name><style>Type A</style></beer>
        <beer><name>Brand B</name><style>Type B</style></beer>
    </root>
    """
    path = tmp_path / "beers.xml"
    path.write_text(xml, encoding="utf-8")

    db = GrafitoDatabase(":memory:")
    results = db.execute(f"""
        WITH '{path.as_uri()}' AS url
        CALL apoc.load.xml(url, './/beer') YIELD value
        RETURN value.name._text AS name, value.style._text AS style
    """)
    assert results == [
        {"name": "Brand A", "style": "Type A"},
        {"name": "Brand B", "style": "Type B"},
    ]
    db.close()


def test_apoc_load_xml_with_compression(tmp_path):
    import bz2
    import gzip
    import lzma
    import zipfile

    xml = "<root><beer><name>Brand A</name></beer></root>"
    xml_path = tmp_path / "beers.xml"
    xml_path.write_text(xml, encoding="utf-8")

    gz_path = tmp_path / "beers.xml.gz"
    gz_path.write_bytes(gzip.compress(xml.encode("utf-8")))

    bz2_path = tmp_path / "beers.xml.bz2"
    bz2_path.write_bytes(bz2.compress(xml.encode("utf-8")))

    xz_path = tmp_path / "beers.xml.xz"
    xz_path.write_bytes(lzma.compress(xml.encode("utf-8")))

    zip_path = tmp_path / "beers.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(xml_path, arcname="data.xml")

    db = GrafitoDatabase(":memory:")
    for path, compression in (
        (gz_path, "gzip"),
        (bz2_path, "bz2"),
        (xz_path, "xz"),
        (zip_path, "zip"),
    ):
        results = db.execute(f"""
            WITH '{path.as_uri()}' AS url
            CALL apoc.load.xml(url, './/beer', {{compression: '{compression}', path: 'data.xml'}})
            YIELD value
            RETURN value.name._text AS name
        """)
        assert results == [{"name": "Brand A"}]
    db.close()


def test_apoc_load_json_params(tmp_path):
    payload = {"items": [{"name": "Brand A"}]}
    path = tmp_path / "beers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    db = GrafitoDatabase(":memory:")
    results = db.execute(f"""
        WITH '{path.as_posix()}' AS url
        CALL apoc.load.jsonParams(
            url,
            {{type: 'lager'}},
            {{"X-Test": 'yes'}},
            {{method: 'GET', payload: {{q: 'x'}}, timeout: 5, retry: 1, failOnError: true}}
        )
        YIELD value
        RETURN value.items[0].name AS name
    """)
    assert results == [{"name": "Brand A"}]
    db.close()


def test_apoc_load_xml_params(tmp_path):
    xml = "<root><beer><name>Brand A</name></beer></root>"
    path = tmp_path / "beers.xml"
    path.write_text(xml, encoding="utf-8")

    db = GrafitoDatabase(":memory:")
    results = db.execute(f"""
        WITH '{path.as_posix()}' AS url
        CALL apoc.load.xmlParams(
            url,
            './/beer',
            {{type: 'lager'}},
            {{"X-Test": 'yes'}},
            {{method: 'GET', payload: {{q: 'x'}}, timeout: 5, retry: 1, failOnError: true}}
        )
        YIELD value
        RETURN value.name._text AS name
    """)
    assert results == [{"name": "Brand A"}]
    db.close()


def test_apoc_load_json_from_tgz(tmp_path):
    import tarfile

    payload = {"person": {"name": "Alice"}}
    json_path = tmp_path / "person.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    tar_path = tmp_path / "testload.tgz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(json_path, arcname="person.json")

    db = GrafitoDatabase(":memory:")
    results = db.execute(f"""
        WITH '{tar_path.as_posix()}!person.json' AS url
        CALL apoc.load.json(url) YIELD value
        RETURN value.person.name AS name
    """)
    assert results == [{"name": "Alice"}]
    db.close()


def test_apoc_load_json_from_tar_bz2_and_xz(tmp_path):
    import tarfile

    payload = {"person": {"name": "Alice"}}
    json_path = tmp_path / "person.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    bz2_path = tmp_path / "testload.tar.bz2"
    with tarfile.open(bz2_path, "w:bz2") as archive:
        archive.add(json_path, arcname="person.json")

    xz_path = tmp_path / "testload.tar.xz"
    with tarfile.open(xz_path, "w:xz") as archive:
        archive.add(json_path, arcname="person.json")

    db = GrafitoDatabase(":memory:")
    for path in (bz2_path, xz_path):
        results = db.execute(f"""
            WITH '{path.as_posix()}!person.json' AS url
            CALL apoc.load.json(url) YIELD value
            RETURN value.person.name AS name
        """)
    assert results == [{"name": "Alice"}]
    db.close()


def test_apoc_import_json_jsonl(tmp_path):
    jsonl = "\n".join(
        [
            json.dumps({
                "type": "node",
                "id": "a",
                "labels": ["Person"],
                "properties": {"name": "Alice"},
            }),
            json.dumps({
                "type": "node",
                "id": "b",
                "labels": ["Person"],
                "properties": {"name": "Bob"},
            }),
            json.dumps({
                "type": "relationship",
                "start": "a",
                "end": "b",
                "label": "KNOWS",
                "properties": {"since": 2020},
            }),
        ]
    )
    path = tmp_path / "people.jsonl"
    path.write_text(jsonl, encoding="utf-8")

    db = GrafitoDatabase(":memory:")
    result = db.execute(f"""
        WITH '{path.as_posix()}' AS url
        CALL apoc.import.json(url) YIELD nodes, relationships
        RETURN nodes, relationships
    """)
    assert result == [{"nodes": 2, "relationships": 1}]

    results = db.execute("""
        MATCH (a:Person)-[r:KNOWS]->(b:Person)
        RETURN a.name AS a, b.name AS b, r.since AS since
    """)
    assert results == [{"a": "Alice", "b": "Bob", "since": 2020}]
    db.close()


def test_apoc_import_json_deflate():
    db = GrafitoDatabase(":memory:")
    result = db.execute("""
        WITH apoc.util.compress(
            '{"type":"node","id":"2","labels":["User"],"properties":{"age":12}}',
            {compression: 'DEFLATE'}
        ) AS jsonCompressed
        CALL apoc.import.json(jsonCompressed, {compression: 'DEFLATE'})
        YIELD nodes, relationships
        RETURN nodes, relationships
    """)
    assert result == [{"nodes": 1, "relationships": 0}]

    results = db.execute("MATCH (n:User) RETURN n.age AS age")
    assert results == [{"age": 12}]
    db.close()


def _vector_db() -> GrafitoDatabase:
    """Two orthogonal unit vectors, so cosine scores are exactly 1.0 or 0.0."""
    db = GrafitoDatabase(':memory:')
    db.create_vector_index("people_vec", dim=2, options={"store_embeddings": True})
    alice = db.create_node(labels=["Person"], properties={"name": "Alice"})
    bob = db.create_node(labels=["Person"], properties={"name": "Bob"})
    db.upsert_embedding(alice.id, [1.0, 0.0], index="people_vec")
    db.upsert_embedding(bob.id, [0.0, 1.0], index="people_vec")
    db.create_relationship(alice.id, bob.id, "KNOWS")
    return db


def test_yield_alias_renames_outputs():
    db = _vector_db()
    results = db.execute("""
        CALL db.vector.search('people_vec', [1.0, 0.0], 1) YIELD node AS person, score AS s
        RETURN person.name AS name, s
    """)
    assert results == [{"name": "Alice", "s": 1.0}]
    db.close()


def test_yield_alias_binds_into_match():
    """The alias must be a real pattern variable, not just a projection name."""
    db = _vector_db()
    results = db.execute("""
        CALL db.vector.search('people_vec', [1.0, 0.0], 1) YIELD node AS a
        MATCH (a)-[:KNOWS]->(b)
        RETURN a.name AS a, b.name AS b
    """)
    assert results == [{"a": "Alice", "b": "Bob"}]
    db.close()


def test_two_calls_with_aliases_bound_to_both_path_ends():
    """txtai-style deep graph search: both endpoints seeded from the ANN index."""
    db = _vector_db()
    results = db.execute("""
        CALL db.vector.search('people_vec', [1.0, 0.0], 1) YIELD node AS a
        CALL db.vector.search('people_vec', [0.0, 1.0], 1) YIELD node AS b
        MATCH p=(a)-[*1..2]->(b)
        RETURN a.name AS ini, b.name AS fin, length(p) AS hops
    """)
    assert results == [{"ini": "Alice", "fin": "Bob", "hops": 1}]
    db.close()


def test_yield_without_alias_keeps_output_names():
    db = _vector_db()
    results = db.execute("""
        CALL db.vector.search('people_vec', [1.0, 0.0], 1) YIELD node, score
        RETURN node.name AS name, score
    """)
    assert results == [{"name": "Alice", "score": 1.0}]
    db.close()


def test_yield_unknown_output_is_rejected():
    db = _vector_db()
    with pytest.raises(Exception, match="does not yield"):
        db.execute("""
            CALL db.vector.search('people_vec', [1.0, 0.0], 1) YIELD nodo
            RETURN nodo
        """)
    db.close()


def test_yield_alias_requires_identifier():
    db = _vector_db()
    with pytest.raises(Exception, match="Expected identifier after AS"):
        db.execute("""
            CALL db.vector.search('people_vec', [1.0, 0.0], 1) YIELD node AS 42
            RETURN node
        """)
    db.close()


def test_map_literal_accepts_keyword_keys():
    """Options maps use reserved words like `index`; the lexer must not block them."""
    db = GrafitoDatabase(':memory:')
    results = db.execute("RETURN {index: 'default', order: 2, limit: 3} AS opts")
    assert results == [{"opts": {"index": "default", "order": 2, "limit": 3}}]
    db.close()
