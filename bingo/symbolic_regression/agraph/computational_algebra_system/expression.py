from .operator_definitions import *


class Expression:

    def __init__(self, operator, operands):
        self._operator = operator
        self._operands = operands
        self._is_constant_valued = None
        self._depends_on = None
        self._hash = None

    @property
    def operator(self):
        return self._operator

    @property
    def operands(self):
        return self._operands

    @property
    def is_constant_valued(self):
        if self._is_constant_valued is None:
            self._is_constant_valued = self._is_derived_from_constants()
        return self._is_constant_valued

    @property
    def depends_on(self):
        if self._depends_on is None:
            self._depends_on = self._find_what_expression_depends_on()
        return self._depends_on

    @property
    def base(self):
        if self._operator == POWER:
            return self._operands[0]
        if self._operator == INTEGER:
            return None
        return self

    @property
    def exponent(self):
        if self._operator == POWER:
            return self._operands[1]
        if self._operator == INTEGER:
            return None
        return Expression(INTEGER, [1, ])

    @property
    def term(self):
        if self._operator == MULTIPLICATION:
            if self._operands[0].is_constant_valued:
                return Expression(MULTIPLICATION, self._operands[1:])
            return self

        if self._operator == INTEGER:
            return None
        return Expression(MULTIPLICATION, [self, ])

    @property
    def coefficient(self):
        if self._operator == MULTIPLICATION and \
                self._operands[0].is_constant_valued:
            return self._operands[0]
        if self._operator == INTEGER:
            return None
        return Expression(INTEGER, [1, ])

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
            return {self.operands[0]}

        return set.union(*[o.depends_on for o in self._operands])

    def map(self, mapped_function):
        mapped_operands = [mapped_function(i) for i in self._operands]
        return Expression(self._operator, mapped_operands)

    def __eq__(self, other):
        if other is None:
            return False
        if hash(self) != hash(other):
            return False
        if self.operator != other.operator:
            return False
        return self.operands == other.operands

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.is_constant_valued or other.is_constant_valued:
            return self._constant_lt(other)

        s_op = self.operator
        o_op = other.operator
        if s_op == MULTIPLICATION or o_op == MULTIPLICATION:
            return self._associative_lt(other, MULTIPLICATION)
        if s_op == POWER or o_op == POWER:
            return self._power_lt(other)
        if s_op == ADDITION or o_op == ADDITION:
            return self._associative_lt(other, ADDITION)
        return self._general_lt(other)

    def _constant_lt(self, other):
        if self.is_constant_valued != other.is_constant_valued:
            return self.is_constant_valued

        return self._general_lt(other)

    def _general_lt(self, other):
        if self.operator != other.operator:
            return self.operator < other.operator
        return self._operands_lt(self.operands, other.operands)

    @staticmethod
    def _operands_lt(s_operands, o_operands):
        for s_operand, o_operand in zip(reversed(s_operands),
                                        reversed(o_operands)):
            if s_operand != o_operand:
                return s_operand < o_operand
        return len(s_operands) < len(o_operands)

    def _associative_lt(self, other, associative_operator):
        if self.operator == associative_operator:
            if other.operator == associative_operator:
                return self._operands_lt(self.operands,
                                         other.operands)
            return self._operands_lt(self.operands, [other, ])
        return self._operands_lt([self, ], other.operands)

    def _power_lt(self, other):
        if self._operator == POWER:
            s_base = self._operands[0]
            s_exponent = self._operands[1]
        else:
            s_base = self
            s_exponent = Expression(INTEGER, [1])

        if other.operator == POWER:
            o_base = other.operands[0]
            o_exponent = other.operands[1]
        else:
            o_base = other
            o_exponent = Expression(INTEGER, [1])

        if s_base == o_base:
            return s_exponent < o_exponent
        return s_base < o_base

    def copy(self):
        return Expression(self._operator, self._operands)

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
            self._hash = hash((self.operator,) +
                              tuple([hash(i) for i in self.operands]))
        return self._hash
