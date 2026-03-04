"""Mathematical expression in tree form for CAS simplification.

This is the internal tree representation used by the simplification
pipeline.  It is *not* the same as :class:`AGraphExpression` — it is a
recursive tree of operator nodes used purely for algebraic manipulation.
"""

from ..operators import (
    POWER,
    INTEGER,
    MULTIPLICATION,
    CONSTANT,
    VARIABLE,
    ADDITION,
    SUBTRACTION,
    DIVISION,
    SAFE_POWER,
    SQUARE,
    CUBE,
    SQRT,
    ABS,
    EXPONENTIAL,
    LOGARITHM,
    SIN,
    COS,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
)

# Sentinel used to distinguish "not yet computed" from a cached ``None``
# value (which is a valid result for INTEGER operands).
_SENTINEL = object()

# ------------------------------------------------------------------ #
#  Interned singletons — allocated once, reused everywhere.           #
# ------------------------------------------------------------------ #
#
# Initialised lazily by _init_singletons() on first access so that
# CASExpression is defined before we try to instantiate them.

_ZERO = None
_ONE = None
_TWO = None
_NEG_ONE = None
_VARIABLE_CACHE = None  # list[CASExpression] for VARIABLE(0)..VARIABLE(7)

# ------------------------------------------------------------------ #
#  Operator sort-key — matches the old symbolic_regression ordering   #
#  so that commutative operand placement is identical.                #
# ------------------------------------------------------------------ #

_OPERATOR_ORDER = {
    INTEGER: -1,
    VARIABLE: 0,
    CONSTANT: 1,
    ADDITION: 2,
    SUBTRACTION: 3,
    MULTIPLICATION: 4,
    DIVISION: 5,
    SIN: 6,
    COS: 7,
    EXPONENTIAL: 8,
    LOGARITHM: 9,
    POWER: 10,
    ABS: 11,
    SQRT: 12,
    SAFE_POWER: 13,
    SINH: 14,
    COSH: 15,
    TAN: 16,
    ARCSIN: 17,
    ARCCOS: 18,
    ARCTAN: 19,
    TANH: 20,
    SQUARE: 21,
    CUBE: 22,
}


def _init_singletons():
    """Bootstrap module-level interned CASExpression singletons."""
    global _ZERO, _ONE, _TWO, _NEG_ONE, _VARIABLE_CACHE  # noqa: PLW0603
    if _ONE is not None:
        return
    _ZERO = CASExpression(INTEGER, [0])
    _ONE = CASExpression(INTEGER, [1])
    _TWO = CASExpression(INTEGER, [2])
    _NEG_ONE = CASExpression(INTEGER, [-1])
    _VARIABLE_CACHE = [CASExpression(VARIABLE, [i]) for i in range(8)]


def get_interned_integer(value):
    """Return a cached CASExpression for common integer values."""
    _init_singletons()
    if value == 0:
        return _ZERO
    if value == 1:
        return _ONE
    if value == 2:
        return _TWO
    if value == -1:
        return _NEG_ONE
    return CASExpression(INTEGER, [value])


def get_interned_variable(index):
    """Return a cached CASExpression for common variable indices."""
    _init_singletons()
    if 0 <= index < len(_VARIABLE_CACHE):
        return _VARIABLE_CACHE[index]
    return CASExpression(VARIABLE, [index])


def _get_one():
    _init_singletons()
    return _ONE


class CASExpression:
    """A mathematical expression in tree form.

    Parameters
    ----------
    operator : int
        Operator ID from :mod:`..operators`.
    operands : list
        Operands — each is either a :class:`CASExpression` or an ``int``
        (for terminal parameter values).
    """

    __slots__ = (
        "_operator",
        "_operands",
        "_is_constant_valued",
        "_depends_on",
        "_hash",
        "_base",
        "_exponent",
        "_term",
        "_coefficient",
    )

    def __init__(self, operator, operands):
        self._operator = operator
        self._operands = operands
        self._is_constant_valued = None
        self._depends_on = None
        self._hash = None
        self._base = _SENTINEL
        self._exponent = _SENTINEL
        self._term = _SENTINEL
        self._coefficient = _SENTINEL

    @property
    def operator(self):
        """The primary mathematical operation in the expression."""
        return self._operator

    @property
    def operands(self):
        """The operands upon which the expression's operator acts."""
        return self._operands

    @property
    def is_constant_valued(self):
        """Whether the expression is derived solely from constant values."""
        if self._is_constant_valued is None:
            self._is_constant_valued = self._is_derived_from_constants()
        return self._is_constant_valued

    @property
    def depends_on(self):
        """The constants, integers and variables in sub-expressions."""
        if self._depends_on is None:
            self._depends_on = self._find_what_expression_depends_on()
        return self._depends_on

    @property
    def base(self):
        """The base *x* in *x^b*."""
        if self._base is _SENTINEL:
            if self._operator == POWER:
                self._base = self._operands[0]
            elif self._operator == INTEGER:
                self._base = None
            else:
                self._base = self
        return self._base

    @property
    def exponent(self):
        """The exponent *b* in *x^b*."""
        if self._exponent is _SENTINEL:
            if self._operator == POWER:
                self._exponent = self._operands[1]
            elif self._operator == INTEGER:
                self._exponent = None
            else:
                self._exponent = _get_one()
        return self._exponent

    @property
    def term(self):
        """The term *x* in *A·x*."""
        if self._term is _SENTINEL:
            if self._operator == MULTIPLICATION:
                if self._operands[0].operator in [INTEGER, CONSTANT]:
                    self._term = CASExpression(MULTIPLICATION, self._operands[1:])
                else:
                    self._term = self
            elif self._operator == INTEGER:
                self._term = None
            else:
                self._term = CASExpression(MULTIPLICATION, [self])
        return self._term

    @property
    def coefficient(self):
        """The coefficient *A* in *A·x*."""
        if self._coefficient is _SENTINEL:
            if self._operator == MULTIPLICATION and self._operands[0].operator in [
                INTEGER,
                CONSTANT,
            ]:
                self._coefficient = self._operands[0]
            elif self._operator == INTEGER:
                self._coefficient = None
            else:
                self._coefficient = _get_one()
        return self._coefficient

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _is_derived_from_constants(self):
        if self._operator in [INTEGER, CONSTANT]:
            return True
        if self._operator == VARIABLE:
            return False
        for operand in self._operands:
            if not operand.is_constant_valued:
                return False
        return True

    def _find_what_expression_depends_on(self):
        if self._operator == INTEGER:
            return {"i"}
        if self._operator == VARIABLE:
            return {"x"}
        if self._operator == CONSTANT:
            return {self._operands[0]}
        return set.union(*[o.depends_on for o in self._operands])

    # ------------------------------------------------------------------ #
    #  Public helpers                                                     #
    # ------------------------------------------------------------------ #

    def map(self, mapped_function):
        """Apply a function to all operands of the expression.

        Returns ``self`` when no operand changed (identity short-circuit).

        Parameters
        ----------
        mapped_function : callable

        Returns
        -------
        CASExpression
        """
        mapped_operands = [mapped_function(i) for i in self._operands]
        if all(m is o for m, o in zip(mapped_operands, self._operands)):
            return self
        return CASExpression(self._operator, mapped_operands)

    def is_zero(self):
        """Whether the expression is the integer ``0``."""
        if self._operator != INTEGER:
            return False
        return self._operands[0] == 0

    def is_one(self):
        """Whether the expression is the integer ``1``."""
        if self._operator != INTEGER:
            return False
        return self._operands[0] == 1

    def same_term(self, other):
        """Check if *self* and *other* share the same additive term.

        Equivalent to ``self.term == other.term`` but avoids allocating
        wrapper ``CASExpression(MULTIPLICATION, [x])`` nodes that the
        ``.term`` property creates for non-MULTIPLICATION operands.

        Returns ``False`` for INTEGER-typed operands (which have term=None).
        """
        s_op = self._operator
        o_op = other._operator

        if s_op == INTEGER or o_op == INTEGER:
            return False

        s_has_coeff = s_op == MULTIPLICATION and self._operands[0].operator in (
            INTEGER,
            CONSTANT,
        )
        o_has_coeff = o_op == MULTIPLICATION and other._operands[0].operator in (
            INTEGER,
            CONSTANT,
        )

        if s_has_coeff and o_has_coeff:
            return self._operands[1:] == other._operands[1:]
        if s_has_coeff and not o_has_coeff:
            s_tail = self._operands[1:]
            # other is also MULTIPLICATION (without leading coeff) →
            # compare tail operands directly.
            if o_op == MULTIPLICATION:
                return list(s_tail) == list(other._operands)
            return len(s_tail) == 1 and s_tail[0] == other
        if not s_has_coeff and o_has_coeff:
            o_tail = other._operands[1:]
            # self is also MULTIPLICATION (without leading coeff) →
            # compare operands directly.
            if s_op == MULTIPLICATION:
                return list(self._operands) == list(o_tail)
            return len(o_tail) == 1 and o_tail[0] == self
        # Neither has a leading coefficient — terms are
        # MULTIPLICATION([self]) vs MULTIPLICATION([other])
        return self == other

    def copy(self):
        """Shallow copy of the expression."""
        return CASExpression(self._operator, self._operands)

    # ------------------------------------------------------------------ #
    #  Comparison / hashing                                               #
    # ------------------------------------------------------------------ #

    def __eq__(self, other):
        if self is other:
            return True
        if other is None:
            return False
        if self._operator != other.operator:
            return False
        return self._operands == other.operands

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.is_constant_valued or other.is_constant_valued:
            return self._constant_lt(other)

        s_op = self._operator
        o_op = other.operator
        if MULTIPLICATION in (s_op, o_op):
            return self._associative_lt(other, MULTIPLICATION)
        if POWER in (s_op, o_op):
            return self._power_lt(other)
        if ADDITION in (s_op, o_op):
            return self._associative_lt(other, ADDITION)
        return self._general_lt(other)

    def _constant_lt(self, other):
        if self.is_constant_valued != other.is_constant_valued:
            return self.is_constant_valued
        return self._general_lt(other)

    def _general_lt(self, other):
        if self._operator != other.operator:
            return _OPERATOR_ORDER[self._operator] < _OPERATOR_ORDER[other.operator]
        return self._operands_lt(self._operands, other.operands)

    @staticmethod
    def _operands_lt(s_operands, o_operands):
        for s_operand, o_operand in zip(reversed(s_operands), reversed(o_operands)):
            if s_operand != o_operand:
                return s_operand < o_operand
        return len(s_operands) < len(o_operands)

    def _associative_lt(self, other, associative_operator):
        if self._operator == associative_operator:
            if other.operator == associative_operator:
                return self._operands_lt(self._operands, other.operands)
            return self._operands_lt(self._operands, [other])
        return self._operands_lt([self], other.operands)

    def _power_lt(self, other):
        s_base = self.base
        s_exponent = self.exponent
        o_base = other.base
        o_exponent = other.exponent

        if s_base == o_base:
            return s_exponent < o_exponent
        return s_base < o_base

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = f"{self._operator}("
        for operand in self._operands:
            string += f"{operand}, "
        string += ")"
        return string

    def __hash__(self):
        if self._hash is None:
            h = hash(self._operator)
            for op in self._operands:
                h = h * 31 ^ hash(op)
            self._hash = h
        return self._hash


# ------------------------------------------------------------------ #
#  Eagerly initialise singletons now that CASExpression is defined.   #
# ------------------------------------------------------------------ #
_init_singletons()
