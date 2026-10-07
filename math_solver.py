import sympy as sp


# ============================================================
# LAYER 1 — SYMBOLIC ALGEBRA & EXPRESSIONS
# ============================================================


def _success(operation, result):
    """Return a standardized successful result."""
    return {
        "success": True,
        "operation": operation,
        "result": result,
    }


def _failure(operation, error):
    """Return a standardized failed result."""
    return {
        "success": False,
        "operation": operation,
        "error": error,
    }


def _parse_expression(expression):
    """
    Convert user-friendly mathematical notation into
    a SymPy expression.

    Example:
        x^2 + 2x + 1
        -> x**2 + 2*x + 1
    """

    expression = expression.strip()

    if not expression:
        raise ValueError("Expression cannot be empty.")

    # Allow ^ as exponent notation.
    expression = expression.replace("^", "**")

    return sp.sympify(
        expression,
        locals={
            "pi": sp.pi,
            "e": sp.E,
            "sqrt": sp.sqrt,
        },
    )


def evaluate_expression(expression):
    """
    Evaluate a mathematical expression.

    Example:
        2 + 5 * 3
        -> 17
    """

    try:
        expr = _parse_expression(expression)

        result = sp.simplify(expr)

        return _success(
            "evaluate",
            result
        )

    except Exception as error:
        return _failure(
            "evaluate",
            str(error)
        )


def simplify_expression(expression):
    """
    Simplify a symbolic mathematical expression.
    """

    try:
        expr = _parse_expression(expression)

        result = sp.simplify(expr)

        return _success(
            "simplify",
            result
        )

    except Exception as error:
        return _failure(
            "simplify",
            str(error)
        )


def expand_expression(expression):
    """
    Expand a symbolic expression.

    Example:
        (x + 2)^2
        -> x^2 + 4*x + 4
    """

    try:
        expr = _parse_expression(expression)

        result = sp.expand(expr)

        return _success(
            "expand",
            result
        )

    except Exception as error:
        return _failure(
            "expand",
            str(error)
        )


def factor_expression(expression):
    """
    Factor a symbolic expression.

    Example:
        x^2 - 5*x + 6
        -> (x - 2)*(x - 3)
    """

    try:
        expr = _parse_expression(expression)

        result = sp.factor(expr)

        return _success(
            "factor",
            result
        )

    except Exception as error:
        return _failure(
            "factor",
            str(error)
        )




# ============================================================
# LAYER 2 — EQUATIONS & SYSTEMS
# ============================================================


def _parse_equation(equation):
    """
    Convert a textual equation into a SymPy Equality.

    Example:
        x + 5 = 12
        -> Eq(x + 5, 12)
    """

    equation = equation.strip()

    if not equation:
        raise ValueError("Equation cannot be empty.")

    if equation.count("=") != 1:
        raise ValueError(
            "Equation must contain exactly one '=' sign."
        )

    left, right = equation.split("=")

    left = left.strip()
    right = right.strip()

    if not left or not right:
        raise ValueError(
            "Both sides of the equation must contain an expression."
        )

    left_expr = _parse_expression(left)
    right_expr = _parse_expression(right)

    return sp.Eq(left_expr, right_expr)


def solve_equation(equation):
    """
    Solve a single mathematical equation.

    Example:
        x + 5 = 12
        -> x = 7
    """

    try:
        eq = _parse_equation(equation)

        variables = eq.free_symbols

        if not variables:
            result = sp.simplify(
                eq.lhs - eq.rhs
            )

            if result == 0:
                return _success(
                    "solve_equation",
                    "Identity: true for all values."
                )

            return _success(
                "solve_equation",
                "Contradiction: no solution."
            )

        solutions = sp.solve(
            eq,
            list(variables)
        )

        return _success(
            "solve_equation",
            solutions
        )

    except Exception as error:
        return _failure(
            "solve_equation",
            str(error)
        )


def solve_system(equations):
    """
    Solve a system of mathematical equations.

    Parameters
    ----------
    equations : list[str]
        Example:
            [
                "x + y = 10",
                "2*x - y = 3"
            ]
    """

    try:
        if not equations:
            raise ValueError(
                "At least one equation is required."
            )

        if not isinstance(equations, (list, tuple)):
            raise TypeError(
                "Equations must be provided as a list or tuple."
            )

        parsed_equations = [
            _parse_equation(equation)
            for equation in equations
        ]

        variables = set()

        for equation in parsed_equations:
            variables.update(
                equation.free_symbols
            )

        if not variables:
            raise ValueError(
                "No variables found in the system."
            )

        solutions = sp.solve(
            parsed_equations,
            sorted(
                variables,
                key=lambda symbol: symbol.name
            ),
            dict=True
        )

        return _success(
            "solve_system",
            solutions
        )

    except Exception as error:
        return _failure(
            "solve_system",
            str(error)
        )