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

# ============================================================
# LAYER 3 — CALCULUS
# ============================================================


def _get_symbol(variable):
    """
    Convert a variable name into a SymPy Symbol.
    """

    variable = variable.strip()

    if not variable:
        raise ValueError(
            "Variable cannot be empty."
        )

    return sp.Symbol(variable)


def differentiate(expression, variable="x", order=1):
    """
    Differentiate an expression.

    Parameters
    ----------
    expression : str
        Mathematical expression.
    variable : str
        Variable with respect to which differentiation is performed.
    order : int
        Order of differentiation.

    Examples
    --------
    differentiate("x^3")
    differentiate("x^2*y + y^3", "x")
    differentiate("x^4", "x", 2)
    """

    try:
        if not isinstance(order, int) or order < 1:
            raise ValueError(
                "Differentiation order must be a positive integer."
            )

        expr = _parse_expression(expression)
        symbol = _get_symbol(variable)

        result = sp.diff(
            expr,
            symbol,
            order
        )

        return _success(
            "differentiate",
            result
        )

    except Exception as error:
        return _failure(
            "differentiate",
            str(error)
        )


def integrate(expression, variable="x"):
    """
    Calculate an indefinite integral.

    Example:
        integrate("x^2")
        -> x^3/3
    """

    try:
        expr = _parse_expression(expression)
        symbol = _get_symbol(variable)

        result = sp.integrate(
            expr,
            symbol
        )

        return _success(
            "integrate",
            result
        )

    except Exception as error:
        return _failure(
            "integrate",
            str(error)
        )


def definite_integral(
    expression,
    variable="x",
    lower=None,
    upper=None
):
    """
    Calculate a definite integral.

    Example:
        ∫(x^2) dx from 0 to 1
        -> 1/3
    """

    try:
        if lower is None or upper is None:
            raise ValueError(
                "Both lower and upper limits are required."
            )

        expr = _parse_expression(expression)
        symbol = _get_symbol(variable)

        lower_bound = sp.sympify(
            str(lower).replace("^", "**")
        )

        upper_bound = sp.sympify(
            str(upper).replace("^", "**")
        )

        result = sp.integrate(
            expr,
            (symbol, lower_bound, upper_bound)
        )

        return _success(
            "definite_integral",
            result
        )

    except Exception as error:
        return _failure(
            "definite_integral",
            str(error)
        )


def calculate_limit(
    expression,
    variable="x",
    point=0,
    direction="both"
):
    """
    Calculate a mathematical limit.

    Examples:
        lim x->0 sin(x)/x
        lim x->oo 1/x
    """

    try:
        expr = _parse_expression(expression)
        symbol = _get_symbol(variable)

        if isinstance(point, str):
            point_value = point.strip().lower()

            if point_value in ("oo", "inf", "infinity"):
                point = sp.oo

            elif point_value in (
                "-oo",
                "-inf",
                "-infinity"
            ):
                point = -sp.oo

            else:
                point = sp.sympify(
                    point.replace("^", "**")
                )

        else:
            point = sp.sympify(point)

        if direction not in (
            "both",
            "+",
            "-"
        ):
            raise ValueError(
                "Direction must be 'both', '+' or '-'."
            )

        if direction == "both":
            result = sp.limit(
                expr,
                symbol,
                point
            )

        else:
            result = sp.limit(
                expr,
                symbol,
                point,
                dir=direction
            )

        return _success(
            "limit",
            result
        )

    except Exception as error:
        return _failure(
            "limit",
            str(error)
        )


def series_expansion(
    expression,
    variable="x",
    point=0,
    order=6
):
    """
    Generate a Taylor/Maclaurin series expansion.

    Example:
        series_expansion("sin(x)")
    """

    try:
        if not isinstance(order, int) or order < 1:
            raise ValueError(
                "Series order must be a positive integer."
            )

        expr = _parse_expression(expression)
        symbol = _get_symbol(variable)

        point_value = sp.sympify(
            str(point).replace("^", "**")
        )

        result = sp.series(
            expr,
            symbol,
            point_value,
            order
        )

        return _success(
            "series",
            result
        )

    except Exception as error:
        return _failure(
            "series",
            str(error)
        )




# ============================================================
# LAYER 4 — LINEAR ALGEBRA
# ============================================================

import numpy as np


# ------------------------------------------------------------
# VECTORS
# ------------------------------------------------------------

def create_vector(values):
    try:
        vector = np.array(values, dtype=float)
        return _success("create_vector", vector)
    except Exception as error:
        return _failure("create_vector", str(error))


def vector_add(vector_a, vector_b):
    try:
        a = np.array(vector_a, dtype=float)
        b = np.array(vector_b, dtype=float)

        if a.shape != b.shape:
            raise ValueError("Vectors must have the same dimensions.")

        result = a + b
        return _success("vector_add", result)
    except Exception as error:
        return _failure("vector_add", str(error))


def vector_subtract(vector_a, vector_b):
    try:
        a = np.array(vector_a, dtype=float)
        b = np.array(vector_b, dtype=float)

        if a.shape != b.shape:
            raise ValueError("Vectors must have the same dimensions.")

        result = a - b
        return _success("vector_subtract", result)
    except Exception as error:
        return _failure("vector_subtract", str(error))


def scalar_multiply_vector(scalar, vector):
    try:
        vector = np.array(vector, dtype=float)
        result = scalar * vector
        return _success("scalar_multiply_vector", result)
    except Exception as error:
        return _failure("scalar_multiply_vector", str(error))


def dot_product(vector_a, vector_b):
    try:
        a = np.array(vector_a, dtype=float)
        b = np.array(vector_b, dtype=float)

        if a.shape != b.shape:
            raise ValueError("Vectors must have the same dimensions.")

        result = np.dot(a, b)
        return _success("dot_product", result)
    except Exception as error:
        return _failure("dot_product", str(error))


def vector_norm(vector):
    try:
        vector = np.array(vector, dtype=float)
        result = np.linalg.norm(vector)
        return _success("vector_norm", result)
    except Exception as error:
        return _failure("vector_norm", str(error))


# ------------------------------------------------------------
# MATRICES
# ------------------------------------------------------------

def create_matrix(values):
    try:
        matrix = np.array(values, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Matrix must be two-dimensional.")

        return _success("create_matrix", matrix)
    except Exception as error:
        return _failure("create_matrix", str(error))


def matrix_add(matrix_a, matrix_b):
    try:
        a = np.array(matrix_a, dtype=float)
        b = np.array(matrix_b, dtype=float)

        if a.shape != b.shape:
            raise ValueError("Matrices must have the same dimensions.")

        result = a + b
        return _success("matrix_add", result)
    except Exception as error:
        return _failure("matrix_add", str(error))


def matrix_multiply(matrix_a, matrix_b):
    try:
        a = np.array(matrix_a, dtype=float)
        b = np.array(matrix_b, dtype=float)

        if a.shape[1] != b.shape[0]:
            raise ValueError(
                "Number of columns in the first matrix must "
                "equal the number of rows in the second matrix."
            )

        result = np.matmul(a, b)
        return _success("matrix_multiply", result)
    except Exception as error:
        return _failure("matrix_multiply", str(error))


def matrix_transpose(matrix):
    try:
        matrix = np.array(matrix, dtype=float)
        result = matrix.T
        return _success("matrix_transpose", result)
    except Exception as error:
        return _failure("matrix_transpose", str(error))


# ------------------------------------------------------------
# MATRIX PROPERTIES
# ------------------------------------------------------------

def matrix_determinant(matrix):
    try:
        matrix = np.array(matrix, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Matrix must be two-dimensional.")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Determinant requires a square matrix.")

        result = np.linalg.det(matrix)
        return _success("matrix_determinant", result)
    except Exception as error:
        return _failure("matrix_determinant", str(error))


def matrix_inverse(matrix):
    try:
        matrix = np.array(matrix, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Matrix must be two-dimensional.")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Inverse requires a square matrix.")

        if np.isclose(np.linalg.det(matrix), 0):
            raise ValueError("Matrix is singular and cannot be inverted.")

        result = np.linalg.inv(matrix)
        return _success("matrix_inverse", result)
    except Exception as error:
        return _failure("matrix_inverse", str(error))


def matrix_rank(matrix):
    try:
        matrix = np.array(matrix, dtype=float)
        result = np.linalg.matrix_rank(matrix)
        return _success("matrix_rank", result)
    except Exception as error:
        return _failure("matrix_rank", str(error))

# ------------------------------------------------------------
# EIGENVALUES AND EIGENVECTORS
# ------------------------------------------------------------

def matrix_eigenvalues(matrix):
    try:
        matrix = np.array(matrix, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Matrix must be two-dimensional.")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Eigenvalues require a square matrix.")

        result = np.linalg.eigvals(matrix)
        return _success("matrix_eigenvalues", result)
    except Exception as error:
        return _failure("matrix_eigenvalues", str(error))


def matrix_eigenvectors(matrix):
    try:
        matrix = np.array(matrix, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Matrix must be two-dimensional.")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Eigenvectors require a square matrix.")

        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        result = {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
        }

        return _success("matrix_eigenvectors", result)
    except Exception as error:
        return _failure("matrix_eigenvectors", str(error))