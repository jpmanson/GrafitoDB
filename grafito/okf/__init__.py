"""High-level Open Knowledge Format (OKF) API for GrafitoDB.

``OKFBundle`` is an OKF-flavored façade over a ``GrafitoDatabase`` graph. Import
a bundle, navigate concepts/links/citations, search by text or meaning, and
export back to markdown — with the full graph one attribute away (``bundle.db``).

```python
from grafito.okf import OKFBundle

kb = OKFBundle.load("examples/okf/okf_knowledge_base")
kb.layers()                       # {'decisions': 3, 'glossary': 3, 'runbooks': 1}
kb.concept("decisions/0001-use-sqlite").links()
kb.search("query performance", type="Playbook")
kb.db.execute("MATCH (n) RETURN count(n)")   # escape hatch
```
"""

from ..filters import PropertyFilter, PropertyFilterGroup
from ..importers.okf import BundleDiff, ConceptDelta, diff_okf_bundles
from ..importers.okf import lint_bundle as lint_okf_bundle
from ..importers.okf import validate_bundle as validate_okf_bundle
from .agent import (
    AgentRun,
    AnthropicChat,
    BundleTools,
    Chat,
    OpenAIChat,
    ThreadConfinedTools,
    ToolCall,
    ToolRegistry,
    ToolSet,
    run_agent,
)
from .bundle import OKFBundle
from .concept import Concept, ContextPack, Hit, Proposal
from .rerank import (
    CohereReranker,
    CrossEncoderReranker,
    JinaReranker,
    LexicalReranker,
    Reranker,
    VoyageReranker,
    concept_text,
)

__all__ = [
    "OKFBundle",
    "validate_okf_bundle",
    "lint_okf_bundle",
    "diff_okf_bundles",
    "BundleDiff",
    "ConceptDelta",
    "BundleTools",
    "ThreadConfinedTools",
    "Chat",
    "ToolSet",
    "ToolRegistry",
    "AnthropicChat",
    "OpenAIChat",
    "run_agent",
    "AgentRun",
    "ToolCall",
    "Concept",
    "ContextPack",
    "Hit",
    "Proposal",
    "PropertyFilter",
    "PropertyFilterGroup",
    "Reranker",
    "LexicalReranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "VoyageReranker",
    "JinaReranker",
    "concept_text",
]
