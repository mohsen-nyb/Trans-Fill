from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from string import ascii_letters, digits, whitespace
import re
from typing import Any, Dict, List, Union


# Inclusive-inclusive interval
POSITION = [-100, 100]
# 0 is intentionally missing
INDEX = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
# Picking the first few for convenience
# TODO: Might want to change this later
DELIMITER = '&,.?!@()[]%{}/:;$#' + whitespace
# Should this be same as Type.CHAR?
CHARACTER = ''.join([ascii_letters, digits, DELIMITER])


class DSL(ABC):
    """Abstract class for all DSL operators."""

    @abstractmethod
    def eval(self, value: str) -> str:
        """Evaluate the DSL operator with the given value."""
        raise NotImplementedError

    @abstractmethod
    def to_string(self, indent: int, tab: int) -> str:
        """Return a string representation of the DSL operator."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.to_string(indent=0, tab=4)

    @abstractmethod
    def to_tokens(self, op_token_table: Dict[Any, int]) -> List[int]:
        """Convert tree of DSL operators into a list of tokens."""
        raise NotImplementedError


class Program(DSL):
    """Abstract class for Program in grammar."""
    pass


class Concat(Program):
    """Concatenates expressions."""

    def __init__(self, *args):
        self.expressions = args

    def eval(self, value):
        return ''.join([
            e.eval(value)
            for e in self.expressions
        ])

    def to_string(self, indent, tab):
        sub_exps = [
            e.to_string(indent=indent+tab, tab=tab)
            for e in self.expressions
        ]
        return op_to_string('Concat', sub_exps, indent, recursive=True)

    def to_tokens(self, op_token_table):
        sub_tokens = [
            e.to_tokens(op_token_table)
            for e in self.expressions
        ]
        return reduce(
            lambda a, b: a + [op_token_table[self.__class__]] + b,
            sub_tokens,
        ) + [op_token_table['EOS']]


class Expression(DSL):
    """Abstract class for Expression in grammar."""
    pass


class Substring(Expression):
    """Abstract class for Substring in grammar."""
    pass


class Nesting(Expression):
    """Abstract class for Nesting in grammar."""
    pass


class Compose(Nesting):
    """Composes two Nesting expressions."""

    def __init__(self, nesting, nesting_or_substring):
        self.nesting = nesting
        self.nesting_or_substring = nesting_or_substring

    def eval(self, value):
        return self.nesting.eval(
            self.nesting_or_substring.eval(value),
        )

    def to_string(self, indent, tab):
        new_indent = indent + tab
        nesting = self.nesting.to_string(indent=new_indent, tab=tab)
        nesting_or_substring = self.nesting_or_substring.to_string(
            indent=new_indent,
            tab=tab,
        )
        return op_to_string(
            'Compose',
            [nesting, nesting_or_substring],
            indent,
            recursive=True)

    def to_tokens(self, op_token_table):
        return (self.nesting.to_tokens(op_token_table)
                + self.nesting_or_substring.to_tokens(op_token_table))


class ConstStr(Expression):
    """A constant string expression."""

    def __init__(self, char):
        self.char = char

    def eval(self, value):
        return self.char

    def to_string(self, indent, tab):
        return op_to_string('ConstStr', [self.char], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[self.__class__], op_token_table[self.char]]


class SubStr(Substring):
    """Returns substring of a string."""

    def __init__(self, pos1, pos2):
        self.pos1 = pos1
        self.pos2 = pos2

    @staticmethod
    def _substr_index(pos, value):
        # DSL index starts at one
        if pos > 0:
            return pos - 1

        # Prevent underflow
        if abs(pos) > len(value):
            return 0

        return pos

    def eval(self, value):
        p1 = SubStr._substr_index(self.pos1, value)
        p2 = SubStr._substr_index(self.pos2, value)

        # Edge case: When p2 == -1, incrementing by one doesn't
        # make it inclusive. Instead, an empty string is always returned.
        if p2 == -1:
            return value[p1:]

        return value[p1:p2+1]

    def to_string(self, indent, tab):
        return op_to_string('SubStr', [self.pos1, self.pos2], indent)

    def to_tokens(self, op_token_table):
        return [
            op_token_table[self.__class__],
            op_token_table[self.pos1],
            op_token_table[self.pos2],
        ]


class GetSpan(Substring):
    """Returns substring between two occurrences of regex matches."""

    def __init__(self, dsl_regex1, index1, bound1, dsl_regex2, index2, bound2):
        self.dsl_regex1 = dsl_regex1
        self.index1 = index1
        self.bound1 = bound1
        self.dsl_regex2 = dsl_regex2
        self.index2 = index2
        self.bound2 = bound2

    # By convention, we always prefix the DSL regex with 'dsl_' as a way to
    # distinguish it with regular regexes.
    @staticmethod
    def _span_index(dsl_regex, index, bound, value):
        matches = match_dsl_regex(dsl_regex, value)
        # Positive indices start at 1, so we need to substract 1
        index = index if index < 0 else index - 1

        if index >= len(matches):
            return len(matches) - 1

        if index < -len(matches):
            return 0

        span = matches[index]
        return span[0] if bound == Boundary.START else span[1]

    def eval(self, value):
        p1 = GetSpan._span_index(
            dsl_regex=self.dsl_regex1,
            index=self.index1,
            bound=self.bound1,
            value=value,
        )
        p2 = GetSpan._span_index(
            dsl_regex=self.dsl_regex2,
            index=self.index2,
            bound=self.bound2,
            value=value,
        )
        return value[p1:p2]

    def to_string(self, indent, tab):
        return op_to_string(
            'GetSpan',
            [
                self.dsl_regex1,
                self.index1,
                self.bound1,
                self.dsl_regex2,
                self.index2,
                self.bound2,
            ],
            indent,
        )

    def to_tokens(self, op_token_table):
        return [
            op_token_table[elem]
            for elem in [
                self.__class__,
                self.dsl_regex1,
                self.index1,
                self.bound1,
                self.dsl_regex2,
                self.index2,
                self.bound2,
            ]
        ]


class GetToken(Nesting):
    """Return the ith token that matches the given type."""

    def __init__(self, type_, index):
        self.type_ = type_
        self.index = index

    def eval(self, value):
        matches = match_type(self.type_, value)
        i = self.index
        if self.index > 0:
            # Positive indices start at 1
            i -= 1
        return matches[i]

    def to_string(self, indent, tab):
        return op_to_string('GetToken', [self.type_, self.index], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.type_, self.index)]]


class ToCase(Nesting):
    """Convert the string to the given case."""

    def __init__(self, case):
        self.case = case

    def eval(self, value):
        if self.case == Case.PROPER:
            return value.capitalize()

        if self.case == Case.ALL_CAPS:
            return value.upper()

        if self.case == Case.LOWER:
            return value.lower()

        raise ValueError('Invalid case: {}'.format(self.case))

    def to_string(self, indent, tab):
        return op_to_string('ToCase', [self.case], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.case)]]


class Replace(Nesting):
    """Replace the given delimiter with another."""

    def __init__(self, delim1, delim2):
        self.delim1 = delim1
        self.delim2 = delim2

    def eval(self, value):
        return value.replace(self.delim1, self.delim2)

    def to_string(self, indent, tab):
        return op_to_string('Replace', [self.delim1, self.delim2], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.delim1, self.delim2)]]


class Trim(Nesting):
    """Trim the given string of whitespace."""

    def eval(self, value):
        return value.strip()

    def to_string(self, indent, tab):
        return op_to_string('Trim', [], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[self.__class__]]


class GetUpto(Nesting):
    """Get the substring that ends at the first occurrence of the regex."""

    def __init__(self, dsl_regex):
        self.dsl_regex = dsl_regex

    def eval(self, value):
        matches = match_dsl_regex(self.dsl_regex, value)

        if len(matches) == 0:
            return ''

        first = matches[0]
        return value[:first[1]]

    def to_string(self, indent, tab):
        return op_to_string('GetUpto', [self.dsl_regex], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.dsl_regex)]]


class GetFrom(Nesting):
    """
    Get the substring that starts at the end of the
    first occurrence of the regex.
    """

    def __init__(self, dsl_regex):
        self.dsl_regex = dsl_regex

    def eval(self, value):
        matches = match_dsl_regex(self.dsl_regex, value)

        if len(matches) == 0:
            return ''

        first = matches[0]
        return value[first[1]:]

    def to_string(self, indent, tab):
        return op_to_string('GetFrom', [self.dsl_regex], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.dsl_regex)]]


class GetFirst(Nesting):
    """
    Get the concatenation of the first i tokens that match
    the given type.
    """

    def __init__(self, type_, index):
        self.type_ = type_
        self.index = index

    def eval(self, value):
        matches = match_type(self.type_, value)

        if self.index < 0:
            raise IndexError

        return ''.join(matches[:self.index])

    def to_string(self, indent, tab):
        return op_to_string('GetFirst', [self.type_, self.index], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.type_, self.index)]]


class GetAll(Nesting):
    """Get the concatenation of all tokens that match the given type."""

    def __init__(self, type_):
        self.type_ = type_

    def eval(self, value):
        return ' '.join(match_type(self.type_, value))

    def to_string(self, indent, tab):
        return op_to_string('GetAll', [self.type_], indent)

    def to_tokens(self, op_token_table):
        return [op_token_table[(self.__class__, self.type_)]]


class Type(Enum):
    """The type of token to match."""
    NUMBER = 1
    WORD = 2
    ALPHANUM = 3
    ALL_CAPS = 4
    PROP_CASE = 5
    LOWER = 6
    DIGIT = 7
    CHAR = 8


class Case(Enum):
    """The case to convert to."""
    PROPER = 1
    ALL_CAPS = 2
    LOWER = 3


class Boundary(Enum):
    """The side of boundary e.g. the start or end of a match."""
    START = 1
    END = 2


def match_type(type_: Type, value: str) -> List[str]:
    """Helper function to return all matches of a given type."""
    regex = regex_for_type(type_)
    return re.findall(regex, value)


def match_dsl_regex(dsl_regex: Union[Type, str], value: str):
    """Match the given DSL regex."""
    if isinstance(dsl_regex, Type):
        regex = regex_for_type(dsl_regex)
    else:
        assert len(dsl_regex) == 1 and dsl_regex in DELIMITER
        regex = '[' + re.escape(dsl_regex) + ']'

    return [
        match.span()
        for match in re.finditer(regex, value)
    ]


def regex_for_type(type_: Type) -> str:
    if type_ == Type.NUMBER:
        return '[0-9]+'

    if type_ == Type.WORD:
        return '[A-Za-z]+'

    if type_ == Type.ALPHANUM:
        return '[A-Za-z0-9]+'

    if type_ == Type.ALL_CAPS:
        return '[A-Z]+'

    if type_ == Type.PROP_CASE:
        return '[A-Z][a-z]+'

    if type_ == Type.LOWER:
        return '[a-z]+'

    if type_ == Type.DIGIT:
        return '[0-9]'

    # TODO: Should this use CHARACTER?
    if type_ == Type.CHAR:
        return '[A-Za-z0-9]'

    raise ValueError('Unsupported type: {}'.format(type_))


def op_to_string(
        name: str,
        raw_args: List,
        indent: int,
        recursive=False) -> str:
    """
    Helper function to convert an operator to a pretty string.

    :param name: The name of the operator.
    :param raw_args: The arguments to the operator.
    :param indent: The number of spaces to indent the string.
    :param recursive: Whether to format and nest the printing to allow for
        args that are operators.
    """
    indent_str = ' ' * indent

    if recursive:
        args_str = ',\n'.join(raw_args)
        return '{indent_str}{name}(\n{args_str}\n{indent_str})'.format(
            indent_str=indent_str,
            name=name,
            args_str=args_str,
        )

    args = [repr(a) for a in raw_args]
    args_str = ', '.join(args)
    return '{indent_str}{name}({args_str})'.format(
        indent_str=indent_str,
        name=name,
        args_str=args_str,
    )
