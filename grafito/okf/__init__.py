"""High-level Open Knowledge Format (OKF) API for GrafitoDB.

``OKFBundle`` is an OKF-flavored façade over a ``GrafitoDatabase`` graph. Import
a bundle, navigate concepts/links/citations, search by text or meaning, and
export back to markdown — with the full graph one attribute away (``bundle.db``).

```python
from grafito.okf import OKFBundle

kb = OKFBundle.load("examples/okf_knowledge_base")
kb.layers()                       # {'decisions': 3, 'glossary': 3, 'runbooks': 1}
kb.concept("decisions/0001-use-sqlite").links()
kb.search("query performance", type="Playbook")
kb.db.execute("MATCH (n) RETURN count(n)")   # escape hatch
```
"""

from .bundle import OKFBundle
from .concept import Concept, ContextPack, Hit
from .rerank import CohereReranker, LexicalReranker, Reranker, concept_text

__all__ = [
    "OKFBundle",
    "Concept",
    "ContextPack",
    "Hit",
    "Reranker",
    "LexicalReranker",
    "CohereReranker",
    "concept_text",
]
