# APOC: XML / HTML

GrafitoDB provides a subset of APOC loading procedures for XML and HTML.
They accept local file paths or HTTP(S) URLs and return rows with a `value` field.

## apoc.load.xml

Load XML from a file or URL and extract matching elements via XPath.

**Signature**

```cypher
CALL apoc.load.xml(source, xpath [, options]) YIELD value
```

**Arguments**

- `source` (string): file path, `file://` URL, or HTTP(S) URL.
- `xpath` (string): XPath for matching elements.
- `options` (map, optional):
  - `compression`: `gzip`, `bz2`, `xz`, `zip` (or `gz`, `bzip2`, `lzma`).
  - `path` / `fileName`: name of the XML file inside a ZIP archive.
  - For HTTP(S) sources, you can also pass: `method`, `payload`, `timeout`,
    `retry`, `failOnError`, `headers`, `auth` (same as `apoc.load.xmlParams`).

**Return**

Each row contains a map derived from the XML element. The map includes:

- `_tag`: element name
- `_attributes`: attributes map (if present)
- `_text`: trimmed text content (if present)
- Nested children using their tag names; repeated tags become lists.

**Example**

```cypher
CALL apoc.load.xml('data/books.xml', '/catalog/book') YIELD value
RETURN value._attributes.id AS id, value.title._text AS title
```

## apoc.load.xmlParams

Load XML with query params, headers, and request options.

**Signature**

```cypher
CALL apoc.load.xmlParams(source, xpath, params, headers [, options]) YIELD value
```

**Arguments**

- `source` (string): file path or HTTP(S) URL.
- `xpath` (string): XPath for matching elements.
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
  - `compression`, `path` / `fileName` (for compressed/ZIP sources)

**Example**

```cypher
CALL apoc.load.xmlParams(
  'https://example.com/feed.xml',
  '/feed/entry',
  {limit: 10},
  {Accept: 'application/xml'},
  {timeout: 5}
) YIELD value
RETURN value.title._text AS title
```

## apoc.load.html

Load HTML and extract text content with a minimal selector engine.

**Signature**

```cypher
CALL apoc.load.html(source, selectors) YIELD value
```

**Arguments**

- `source` (string): file path or HTTP(S) URL.
- `selectors` (map): `alias -> selector` pairs.

**Selector syntax**

GrafitoDB implements a **minimal** descendant selector:

- Tag + optional classes: `div.card`, `a.link`
- Descendant chains: `div.card h2`
- Optional `:eq(n)` filter per segment: `li:eq(0)`

**Return**

`value` is a map where each key is the selector alias and each value is a list
of `{text: ...}` entries for matched nodes.

**Example**

```cypher
CALL apoc.load.html(
  'https://example.com',
  {titles: 'article h2', links: 'article a.link'}
) YIELD value
RETURN value.titles
```

## Notes

- XML supports automatic compression detection by file extension (`.gz`, `.bz2`,
  `.xz`, `.zip`).
- For ZIP XML, use `options.path` or `options.fileName` to pick the entry.
- HTML extraction returns text only (no attributes).
