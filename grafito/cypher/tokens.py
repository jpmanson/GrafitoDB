"""Token definitions for Cypher lexer."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    """Token types for Cypher queries."""

    # Keywords
    CREATE = auto()
    MATCH = auto()
    MERGE = auto()
    DROP = auto()
    IF = auto()
    EXISTS = auto()
    UNIQUE = auto()
    CONSTRAINT = auto()
    CONSTRAINTS = auto()
    REQUIRE = auto()
    ASSERT = auto()
    FOREACH = auto()
    WHERE = auto()
    RETURN = auto()
    DELETE = auto()
    SET = auto()
    REMOVE = auto()
    ORDER = auto()
    BY = auto()
    FOR = auto()
    INDEX = auto()
    INDEXES = auto()
    SHOW = auto()
    NODE = auto()
    RELATIONSHIP = auto()
    LIMIT = auto()
    SKIP = auto()
    ASC = auto()
    DESC = auto()
    OPTIONAL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    AS = auto()
    WITH = auto()
    ON = auto()
    UNION = auto()
    ALL = auto()
    CALL = auto()
    YIELD = auto()
    DISTINCT = auto()
    UNWIND = auto()
    COLLECT = auto()
    STDDEV = auto()
    PERCENTILECONT = auto()
    IS = auto()
    SHORTESTPATH = auto()
    ALLSHORTESTPATHS = auto()
    IN = auto()
    ANY = auto()
    NONE = auto()
    SINGLE = auto()
    CASE = auto()
    WHEN = auto()
    THEN = auto()
    ELSE = auto()
    END = auto()
    LOAD = auto()
    CSV = auto()
    HEADERS = auto()
    FROM = auto()

    # Aggregation functions
    COUNT = auto()
    SUM = auto()
    AVG = auto()
    MIN = auto()
    MAX = auto()

    # Symbols
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    LBRACE = auto()       # {
    RBRACE = auto()       # }
    LBRACKET = auto()     # [
    RBRACKET = auto()     # ]
    COLON = auto()        # :
    COMMA = auto()        # ,
    DOT = auto()          # .
    ASTERISK = auto()     # *
    PLUS = auto()         # +
    PIPE = auto()         # |
    ARROW_RIGHT = auto()  # ->
    ARROW_LEFT = auto()   # <-
    DASH = auto()         # -

    # Comparison operators
    EQ = auto()           # =
    NEQ = auto()          # !=
    LT = auto()           # <
    GT = auto()           # >
    LTE = auto()          # <=
    GTE = auto()          # >=

    # Literals
    IDENTIFIER = auto()
    PARAMETER = auto()    # $name
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()      # true, false
    NULL = auto()         # null

    # Special
    EOF = auto()          # End of file
    NEWLINE = auto()      # \n


@dataclass
class Token:
    """Represents a token in a Cypher query."""

    type: TokenType
    value: Any
    line: int = 1
    column: int = 1

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


# Keywords mapping (case-insensitive)
KEYWORDS = {
    'create': TokenType.CREATE,
    'match': TokenType.MATCH,
    'merge': TokenType.MERGE,
    'drop': TokenType.DROP,
    'if': TokenType.IF,
    'exists': TokenType.EXISTS,
    'unique': TokenType.UNIQUE,
    'constraint': TokenType.CONSTRAINT,
    'constraints': TokenType.CONSTRAINTS,
    'require': TokenType.REQUIRE,
    'assert': TokenType.ASSERT,
    'foreach': TokenType.FOREACH,
    'where': TokenType.WHERE,
    'return': TokenType.RETURN,
    'delete': TokenType.DELETE,
    'set': TokenType.SET,
    'remove': TokenType.REMOVE,
    'order': TokenType.ORDER,
    'by': TokenType.BY,
    'for': TokenType.FOR,
    'index': TokenType.INDEX,
    'indexes': TokenType.INDEXES,
    'show': TokenType.SHOW,
    'node': TokenType.NODE,
    'relationship': TokenType.RELATIONSHIP,
    'limit': TokenType.LIMIT,
    'skip': TokenType.SKIP,
    'asc': TokenType.ASC,
    'desc': TokenType.DESC,
    'optional': TokenType.OPTIONAL,
    'and': TokenType.AND,
    'or': TokenType.OR,
    'not': TokenType.NOT,
    'as': TokenType.AS,
    'with': TokenType.WITH,
    'on': TokenType.ON,
    'union': TokenType.UNION,
    'all': TokenType.ALL,
    'call': TokenType.CALL,
    'yield': TokenType.YIELD,
    'distinct': TokenType.DISTINCT,
    'unwind': TokenType.UNWIND,
    'collect': TokenType.COLLECT,
    'stddev': TokenType.STDDEV,
    'percentilecont': TokenType.PERCENTILECONT,
    'is': TokenType.IS,
    'shortestpath': TokenType.SHORTESTPATH,
    'allshortestpaths': TokenType.ALLSHORTESTPATHS,
    'in': TokenType.IN,
    'any': TokenType.ANY,
    'none': TokenType.NONE,
    'single': TokenType.SINGLE,
    'any': TokenType.ANY,
    'none': TokenType.NONE,
    'single': TokenType.SINGLE,
    'case': TokenType.CASE,
    'when': TokenType.WHEN,
    'then': TokenType.THEN,
    'else': TokenType.ELSE,
    'end': TokenType.END,
    'load': TokenType.LOAD,
    'csv': TokenType.CSV,
    'headers': TokenType.HEADERS,
    'from': TokenType.FROM,
    'count': TokenType.COUNT,
    'sum': TokenType.SUM,
    'avg': TokenType.AVG,
    'min': TokenType.MIN,
    'max': TokenType.MAX,
    'true': TokenType.BOOLEAN,
    'false': TokenType.BOOLEAN,
    'null': TokenType.NULL,
}
