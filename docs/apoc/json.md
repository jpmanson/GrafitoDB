# APOC: JSON / JSONL

GrafitoDB provides JSON loading and import procedures inspired by APOC.

## apoc.load.json

Load a JSON object or array from a file or URL.

**Signature**

```cypher
CALL apoc.load.json(source) YIELD value
```

**Arguments**

- `source` (string): file path, `file://` URL, or HTTP(S) URL.
- You can also read from tar archives by using `path!member.json`.

**Return**

- If the JSON is an object, returns a single row with `value` = object.
- If the JSON is an array, returns one row per element.

**Example**

```cypher
CALL apoc.load.json('data/users.json') YIELD value
RETURN value.name, value.email
```

## apoc.load.jsonArray

Ensure the JSON payload is an array.

**Signature**

```cypher
CALL apoc.load.jsonArray(source) YIELD value
```

If the payload is not an array, the procedure raises an error.

## apoc.load.jsonParams

Load JSON with query params, headers, and request options.

**Signature**

```cypher
CALL apoc.load.jsonParams(source, params, headers [, options]) YIELD value
```

**Arguments**

- `source` (string): file path or HTTP(S) URL.
- `params` (map): query parameters to append to the URL.
- `headers` (map): HTTP headers.
- `options` (map, optional):
  - `method` (string, default `GET`)
  - `payload` (string | bytes | map | list)
  - `timeout` (number)
  - `retry` (integer)
  - `failOnError` (boolean, default `true`)
  - `headers` (map) additional headers
  - `auth` (string or `{user, password}` map)

**Example**

```cypher
CALL apoc.load.jsonParams(
  'https://api.example.com/users',
  {limit: 10},
  {Accept: 'application/json'},
  {timeout: 5, retry: 2}
) YIELD value
RETURN value
```

## Caching (HTTP GET only)

If `GRAFITO_APOC_CACHE_DIR` is set and the request is a simple GET (no payload,
headers, or auth), GrafitoDB caches the response body on disk and reuses it.

```bash
export GRAFITO_APOC_CACHE_DIR="/tmp/grafito-apoc-cache"
```

## apoc.import.json

Import nodes and relationships from JSON or JSONL.

**Signature**

```cypher
CALL apoc.import.json(source [, options]) YIELD nodes, relationships
```

**Arguments**

- `source` (string | bytes): file path, URL, or raw bytes.
- `options` (map, optional):
  - `compression`: `DEFLATE`, `GZIP`, `BZ2`, `XZ`, `ZIP`
  - `path`: zip entry name (when `compression = "ZIP"`)
  - `idField` (default `id`)
  - `labelsField` (default `labels`)
  - `propertiesField` (default `properties`)
  - `relTypeField` (default `label`)
  - `startField` (default `start`)
  - `endField` (default `end`)
  - `typeField` (default `type`)

**Input Formats**

GrafitoDB accepts:

1) JSON array of entries
2) JSON object with `{nodes: [...], relationships: [...]}`
3) JSONL (one JSON object per line)

Entries are treated as **nodes** unless one of the following is true:

- `typeField` is set to `relationship` / `rel` / `edge`
- `startField` or `endField` is present

**Node entry example**

```json
{"id": "u1", "labels": ["User"], "properties": {"name": "Ada"}}
```

**Relationship entry example**

```json
{"type": "relationship", "label": "FOLLOWS", "start": "u1", "end": "u2"}
```

**Example import**

```cypher
CALL apoc.import.json('data/graph.json') YIELD nodes, relationships
RETURN nodes, relationships
```

## Notes

- Relationship references must point to existing node IDs in the same import.
- Invalid JSON raises an error; JSONL is parsed line by line.
