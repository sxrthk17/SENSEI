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