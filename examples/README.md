# Grafito examples

Los ejemplos están organizados por tema. Salvo que se indique lo contrario,
ejecútalos desde la raíz del repositorio (`python examples/<carpeta>/<script>.py`).

## `basics/`
Uso fundamental de la base y de Cypher.
- `basic_usage.py` — operaciones CRUD básicas sobre el grafo.
- `advanced_queries.py` — consultas Cypher más elaboradas.
- `cypher_usage.py` — recorrido por la sintaxis Cypher soportada.
- `cypher_persistence.py` — persistir y reabrir una base SQLite.

## `datasets/`
Conjuntos de datos de ejemplo e importadores (cada script lleva su `.cypher`/`.xml`/`.jsonl`/`.dump`).
- `social_network.py`, `company_structure.py`, `northwind.py` (+ `northwind.cypher`).
- `got_import.py` (+ `got-import.cypher`) — Game of Thrones.
- `belgian_beers_import.py` / `belgian_beers_xml_import.py` (+ `.cypher`/`.xml`).
- `import_jsonl.py` (+ `import_jsonl.cypher`, `people.jsonl`).
- `neo4j_dump_import.py` (+ `*.dump`) — importar volcados de Neo4j.

## `visualization/`
Exportación y visualización del grafo.
- `cytoscape_visualize.py`, `d3_visualize.py`, `graphviz_visualize.py`,
  `matplotlib_viz_example.py`, `pyvis_visualize.py`, `networkx_export.py`.

## `rdf/`
Exportación a RDF y ontologías.
- `rdf_export.py`, `rdf_export_custom_ns.py`, `rdf_export_typed.py`, `rdf_ontology_example.py`.

## `okf/`
Open Knowledge Format (façade `OKFBundle`).
- `okf_import.py` (+ bundle tabular `okf_bundle/`).
- `okf_knowledge_base.py` (+ base de conocimiento `okf_knowledge_base/`).

## `semantic/`
Búsqueda semántica e indexadores de texto.
- `document_chunking.py` — `DocumentIngestor`: markdown largo → secciones/pasajes, search, expand/pack, hybrid RRF (sin deps opcionales).
- `pdf_chunking_colab.ipynb` — **Colab**: PDF → chunking → búsqueda con visualizaciones PyVis paso a paso (estudiantes).
- `semantic_faiss_hf.py`, `semantic.ipynb`, `benchmark_text_indexers.py`.
