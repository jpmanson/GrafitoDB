"""Lexer for tokenizing Cypher queries."""

from typing import Optional
from .tokens import Token, TokenType, KEYWORDS
from .exceptions import CypherSyntaxError


class Lexer:
    """Tokenizes Cypher query strings into tokens."""

    def __init__(self, query: str):
        self.query = query
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []

    def current_char(self) -> Optional[str]:
        """Get the current character without advancing."""
        if self.pos >= len(self.query):
            return None
        return self.query[self.pos]

    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Peek ahead at a character without advancing."""
        pos = self.pos + offset
        if pos >= len(self.query):
            return None
        return self.query[pos]

    def advance(self) -> None:
        """Move to the next character, tracking line and column."""
        if self.pos < len(self.query):
            if self.query[self.pos] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1

    def skip_whitespace(self) -> None:
        """Skip whitespace characters (except newlines for now)."""
        while self.current_char() and self.current_char() in ' \t\r\n':
            self.advance()

    def read_string(self, quote: str) -> str:
        """Read a string literal enclosed in quotes."""
        value = ''
        start_line = self.line
        start_column = self.column
        self.advance()  # Skip opening quote

        while self.current_char() and self.current_char() != quote:
            if self.current_char() == '\\':
                self.advance()
                # Handle escape sequences
                if self.current_char() in ('"', "'", '\\'):
                    value += self.current_char()
                    self.advance()
                elif self.current_char() == 'n':
                    value += '\n'
                    self.advance()
                elif self.current_char() == 't':
                    value += '\t'
                    self.advance()
                else:
                    value += self.current_char() or ''
                    self.advance()
            else:
                value += self.current_char()
                self.advance()

        if self.current_char() != quote:
            raise CypherSyntaxError(
                f"Unterminated string literal",
                start_line,
                start_column
            )

        self.advance()  # Skip closing quote
        return value

    def read_number(self) -> Token:
        """Read a number (integer or float)."""
        start_column = self.column
        value = ''
        is_float = False
        has_exponent = False

        while self.current_char():
            char = self.current_char()
            if char.isdigit():
                value += char
                self.advance()
                continue
            if char == '.':
                if self.peek_char() == '.':
                    break
                if is_float:
                    break
                is_float = True
                value += char
                self.advance()
                continue
            if char in ('e', 'E'):
                if has_exponent:
                    break
                has_exponent = True
                is_float = True
                value += char
                self.advance()
                if self.current_char() in ('+', '-'):
                    value += self.current_char()
                    self.advance()
                continue
            break

        if is_float:
            return Token(TokenType.FLOAT, float(value), self.line, start_column)
        else:
            return Token(TokenType.INTEGER, int(value), self.line, start_column)

    def read_parameter(self) -> Token:
        """Read a query parameter reference: $name."""
        start_column = self.column
        self.advance()  # consume '$'
        name = ''
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            name += self.current_char()
            self.advance()
        if not name:
            raise CypherSyntaxError(
                "Expected a parameter name after '$'",
                self.line,
                start_column
            )
        return Token(TokenType.PARAMETER, name, self.line, start_column)

    def read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start_column = self.column
        value = ''

        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            value += self.current_char()
            self.advance()

        # Check if it's a keyword
        lower_value = value.lower()
        if lower_value in KEYWORDS:
            token_type = KEYWORDS[lower_value]
            # For boolean keywords, store the actual boolean value
            if token_type == TokenType.BOOLEAN:
                return Token(token_type, lower_value == 'true', self.line, start_column)
            elif token_type == TokenType.NULL:
                return Token(token_type, None, self.line, start_column)
            else:
                return Token(token_type, value, self.line, start_column)
        else:
            return Token(TokenType.IDENTIFIER, value, self.line, start_column)

    def tokenize(self) -> list[Token]:
        """Tokenize the entire query and return list of tokens."""
        self.tokens = []

        while self.current_char():
            self.skip_whitespace()

            if not self.current_char():
                break

            char = self.current_char()
            start_column = self.column

            # String literals
            if char in ('"', "'"):
                value = self.read_string(char)
                self.tokens.append(Token(TokenType.STRING, value, self.line, start_column))

            # Numbers
            elif char.isdigit():
                self.tokens.append(self.read_number())

            # Identifiers and keywords
            elif char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())

            # Query parameters: $name
            elif char == '$':
                self.tokens.append(self.read_parameter())

            # Two-character operators
            elif char == '-':
                if self.peek_char() == '>':
                    self.tokens.append(Token(TokenType.ARROW_RIGHT, '->', self.line, start_column))
                    self.advance()
                    self.advance()
                else:
                    self.tokens.append(Token(TokenType.DASH, '-', self.line, start_column))
                    self.advance()

            elif char == '<':
                if self.peek_char() == '-':
                    self.tokens.append(Token(TokenType.ARROW_LEFT, '<-', self.line, start_column))
                    self.advance()
                    self.advance()
                elif self.peek_char() == '>':
                    self.tokens.append(Token(TokenType.NEQ, '<>', self.line, start_column))
                    self.advance()
                    self.advance()
                elif self.peek_char() == '=':
                    self.tokens.append(Token(TokenType.LTE, '<=', self.line, start_column))
                    self.advance()
                    self.advance()
                else:
                    self.tokens.append(Token(TokenType.LT, '<', self.line, start_column))
                    self.advance()

            elif char == '>':
                if self.peek_char() == '=':
                    self.tokens.append(Token(TokenType.GTE, '>=', self.line, start_column))
                    self.advance()
                    self.advance()
                else:
                    self.tokens.append(Token(TokenType.GT, '>', self.line, start_column))
                    self.advance()

            elif char == '!':
                if self.peek_char() == '=':
                    self.tokens.append(Token(TokenType.NEQ, '!=', self.line, start_column))
                    self.advance()
                    self.advance()
                else:
                    raise CypherSyntaxError(
                        f"Unexpected character '!' (use '!=' for inequality)",
                        self.line,
                        start_column
                    )

            # Single-character symbols
            elif char == '(':
                self.tokens.append(Token(TokenType.LPAREN, '(', self.line, start_column))
                self.advance()
            elif char == ')':
                self.tokens.append(Token(TokenType.RPAREN, ')', self.line, start_column))
                self.advance()
            elif char == '{':
                self.tokens.append(Token(TokenType.LBRACE, '{', self.line, start_column))
                self.advance()
            elif char == '}':
                self.tokens.append(Token(TokenType.RBRACE, '}', self.line, start_column))
                self.advance()
            elif char == '[':
                self.tokens.append(Token(TokenType.LBRACKET, '[', self.line, start_column))
                self.advance()
            elif char == ']':
                self.tokens.append(Token(TokenType.RBRACKET, ']', self.line, start_column))
                self.advance()
            elif char == ':':
                self.tokens.append(Token(TokenType.COLON, ':', self.line, start_column))
                self.advance()
            elif char == ',':
                self.tokens.append(Token(TokenType.COMMA, ',', self.line, start_column))
                self.advance()
            elif char == '.':
                self.tokens.append(Token(TokenType.DOT, '.', self.line, start_column))
                self.advance()
            elif char == '*':
                self.tokens.append(Token(TokenType.ASTERISK, '*', self.line, start_column))
                self.advance()
            elif char == '+':
                self.tokens.append(Token(TokenType.PLUS, '+', self.line, start_column))
                self.advance()
            elif char == '|':
                self.tokens.append(Token(TokenType.PIPE, '|', self.line, start_column))
                self.advance()
            elif char == '=':
                self.tokens.append(Token(TokenType.EQ, '=', self.line, start_column))
                self.advance()

            else:
                raise CypherSyntaxError(
                    f"Unexpected character '{char}'",
                    self.line,
                    start_column
                )

        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens
