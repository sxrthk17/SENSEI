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


# ============================================================
# LAYER 5 — PROBABILITY + STATISTICS
# ============================================================

import scipy.stats as stats


# ------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

def calculate_mean(values):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        result = np.mean(values)

        return _success("mean", result)

    except Exception as error:
        return _failure("mean", str(error))


def calculate_median(values):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        result = np.median(values)

        return _success("median", result)

    except Exception as error:
        return _failure("median", str(error))


def calculate_mode(values):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        result = stats.mode(values, keepdims=False)

        return _success(
            "mode",
            result.mode
        )

    except Exception as error:
        return _failure("mode", str(error))


def calculate_variance(values, sample=False):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if sample:
            if values.size < 2:
                raise ValueError(
                    "Sample variance requires at least two values."
                )

            result = np.var(values, ddof=1)

        else:
            result = np.var(values)

        return _success("variance", result)

    except Exception as error:
        return _failure("variance", str(error))


def calculate_standard_deviation(values, sample=False):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if sample:
            if values.size < 2:
                raise ValueError(
                    "Sample standard deviation requires at least two values."
                )

            result = np.std(values, ddof=1)

        else:
            result = np.std(values)

        return _success("standard_deviation", result)

    except Exception as error:
        return _failure("standard_deviation", str(error))


def calculate_percentile(values, percentile):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if not 0 <= percentile <= 100:
            raise ValueError(
                "Percentile must be between 0 and 100."
            )

        result = np.percentile(values, percentile)

        return _success("percentile", result)

    except Exception as error:
        return _failure("percentile", str(error))


def calculate_covariance(values_a, values_b):
    try:
        a = np.array(values_a, dtype=float)
        b = np.array(values_b, dtype=float)

        if a.size == 0 or b.size == 0:
            raise ValueError("Values cannot be empty.")

        if a.size != b.size:
            raise ValueError(
                "Both datasets must contain the same number of values."
            )

        result = np.cov(a, b)

        return _success("covariance", result)

    except Exception as error:
        return _failure("covariance", str(error))

    # ------------------------------------------------------------
# STATISTICAL VISUALIZATION
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(values, bins=10):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        fig, ax = plt.subplots()

        ax.hist(values, bins=bins)
        ax.set_title("Distribution of Values")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

        fig.tight_layout()

        return _success("histogram", fig)

    except Exception as error:
        return _failure("histogram", str(error))


def plot_boxplot(values):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        fig, ax = plt.subplots()

        ax.boxplot(values)
        ax.set_title("Box Plot")
        ax.set_ylabel("Value")

        fig.tight_layout()

        return _success("boxplot", fig)

    except Exception as error:
        return _failure("boxplot", str(error))


def plot_scatter(values_x, values_y):
    try:
        x = np.array(values_x, dtype=float)
        y = np.array(values_y, dtype=float)

        if x.size == 0 or y.size == 0:
            raise ValueError("Values cannot be empty.")

        if x.size != y.size:
            raise ValueError(
                "Both datasets must contain the same number of values."
            )

        fig, ax = plt.subplots()

        ax.scatter(x, y)
        ax.set_title("Scatter Plot")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        fig.tight_layout()

        return _success("scatter_plot", fig)

    except Exception as error:
        return _failure("scatter_plot", str(error))


def plot_distribution_seaborn(values):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        fig, ax = plt.subplots()

        sns.histplot(
            values,
            kde=True,
            ax=ax
        )

        ax.set_title("Distribution with Density")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density / Frequency")

        fig.tight_layout()

        return _success("seaborn_distribution", fig)

    except Exception as error:
        return _failure("seaborn_distribution", str(error))

def calculate_probability(favorable, total):
    try:
        if total <= 0:
            raise ValueError("Total outcomes must be greater than zero.")

        if favorable < 0:
            raise ValueError("Favorable outcomes cannot be negative.")

        if favorable > total:
            raise ValueError(
                "Favorable outcomes cannot exceed total outcomes."
            )

        result = favorable / total

        return _success("probability", result)

    except Exception as error:
        return _failure("probability", str(error))


def calculate_conditional_probability(intersection, condition):
    try:
        if condition <= 0:
            raise ValueError(
                "Condition probability must be greater than zero."
            )

        if intersection < 0:
            raise ValueError(
                "Intersection probability cannot be negative."
            )

        if intersection > condition:
            raise ValueError(
                "Intersection probability cannot exceed condition probability."
            )

        result = intersection / condition

        return _success("conditional_probability", result)

    except Exception as error:
        return _failure("conditional_probability", str(error))


def calculate_bayes_probability(
    prior_probability,
    evidence_given_event,
    evidence_given_not_event
):
    try:
        if not 0 <= prior_probability <= 1:
            raise ValueError("Prior probability must be between 0 and 1.")

        if not 0 <= evidence_given_event <= 1:
            raise ValueError(
                "Evidence probability given event must be between 0 and 1."
            )

        if not 0 <= evidence_given_not_event <= 1:
            raise ValueError(
                "Evidence probability given not-event must be between 0 and 1."
            )

        evidence_probability = (
            evidence_given_event * prior_probability
            + evidence_given_not_event * (1 - prior_probability)
        )

        if evidence_probability == 0:
            raise ValueError(
                "Evidence probability is zero, so Bayes probability is undefined."
            )

        result = (
            evidence_given_event * prior_probability
        ) / evidence_probability

        return _success("bayes_probability", result)

    except Exception as error:
        return _failure("bayes_probability", str(error))


# ------------------------------------------------------------
# RANDOM VARIABLES — PMF / PDF / CDF
# ------------------------------------------------------------

def calculate_pmf(probabilities):
    """
    Calculate/validate a Probability Mass Function (PMF).

    PMF is used for discrete random variables.
    The probabilities must be non-negative and sum to 1.
    """
    try:
        probabilities = np.array(probabilities, dtype=float)

        if probabilities.size == 0:
            raise ValueError("Probabilities cannot be empty.")

        if np.any(probabilities < 0):
            raise ValueError("PMF probabilities cannot be negative.")

        if not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError("PMF probabilities must sum to 1.")

        return _success("pmf", probabilities)

    except Exception as error:
        return _failure("pmf", str(error))


def calculate_pdf(values, mean=0, standard_deviation=1):
    """
    Calculate the Probability Density Function of a normal
    continuous random variable.
    """
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if standard_deviation <= 0:
            raise ValueError(
                "Standard deviation must be greater than zero."
            )

        result = stats.norm.pdf(
            values,
            loc=mean,
            scale=standard_deviation
        )

        return _success("pdf", result)

    except Exception as error:
        return _failure("pdf", str(error))


def calculate_cdf(values, mean=0, standard_deviation=1):
    """
    Calculate the Cumulative Distribution Function of a normal
    continuous random variable.
    """
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if standard_deviation <= 0:
            raise ValueError(
                "Standard deviation must be greater than zero."
            )

        result = stats.norm.cdf(
            values,
            loc=mean,
            scale=standard_deviation
        )

        return _success("cdf", result)

    except Exception as error:
        return _failure("cdf", str(error))


# ------------------------------------------------------------
# UNIFORM DISTRIBUTION
# ------------------------------------------------------------

def uniform_pdf(values, lower=0, upper=1):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound.")

        result = stats.uniform.pdf(
            values,
            loc=lower,
            scale=upper - lower
        )

        return _success("uniform_pdf", result)

    except Exception as error:
        return _failure("uniform_pdf", str(error))


def uniform_cdf(values, lower=0, upper=1):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound.")

        result = stats.uniform.cdf(
            values,
            loc=lower,
            scale=upper - lower
        )

        return _success("uniform_cdf", result)

    except Exception as error:
        return _failure("uniform_cdf", str(error))


def uniform_probability(lower_bound, upper_bound, lower=0, upper=1):
    try:
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound.")

        if lower_bound > upper_bound:
            raise ValueError(
                "Probability interval lower bound cannot exceed upper bound."
            )

        if lower_bound < lower or upper_bound > upper:
            raise ValueError(
                "Probability interval must lie within the uniform distribution."
            )

        result = (
            (upper_bound - lower_bound)
            / (upper - lower)
        )

        return _success("uniform_probability", result)

    except Exception as error:
        return _failure("uniform_probability", str(error))


def uniform_mean(lower=0, upper=1):
    try:
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound.")

        result = (lower + upper) / 2

        return _success("uniform_mean", result)

    except Exception as error:
        return _failure("uniform_mean", str(error))


def uniform_variance(lower=0, upper=1):
    try:
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound.")

        result = ((upper - lower) ** 2) / 12

        return _success("uniform_variance", result)

    except Exception as error:
        return _failure("uniform_variance", str(error))


def plot_uniform_distribution(lower=0, upper=1, points=200):
    try:
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound.")

        if points < 2:
            raise ValueError("Points must be at least 2.")

        x = np.linspace(lower, upper, points)
        y = stats.uniform.pdf(
            x,
            loc=lower,
            scale=upper - lower
        )

        fig, ax = plt.subplots()

        ax.plot(x, y)
        ax.set_title("Uniform Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

        fig.tight_layout()

        return _success("uniform_distribution_plot", fig)

    except Exception as error:
        return _failure("uniform_distribution_plot", str(error))


# ------------------------------------------------------------
# BERNOULLI DISTRIBUTION
# ------------------------------------------------------------

def bernoulli_pmf(values, probability=0.5):
    try:
        values = np.array(values, dtype=int)

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        if np.any((values != 0) & (values != 1)):
            raise ValueError(
                "Bernoulli values must be either 0 or 1."
            )

        result = stats.bernoulli.pmf(values, probability)

        return _success("bernoulli_pmf", result)

    except Exception as error:
        return _failure("bernoulli_pmf", str(error))


def bernoulli_cdf(values, probability=0.5):
    try:
        values = np.array(values, dtype=float)

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        result = stats.bernoulli.cdf(values, probability)

        return _success("bernoulli_cdf", result)

    except Exception as error:
        return _failure("bernoulli_cdf", str(error))


def bernoulli_probability(value, probability=0.5):
    try:
        if value not in (0, 1):
            raise ValueError("Bernoulli value must be either 0 or 1.")

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        result = stats.bernoulli.pmf(value, probability)

        return _success("bernoulli_probability", result)

    except Exception as error:
        return _failure("bernoulli_probability", str(error))


def bernoulli_mean(probability=0.5):
    try:
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        return _success("bernoulli_mean", probability)

    except Exception as error:
        return _failure("bernoulli_mean", str(error))


def bernoulli_variance(probability=0.5):
    try:
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        result = probability * (1 - probability)

        return _success("bernoulli_variance", result)

    except Exception as error:
        return _failure("bernoulli_variance", str(error))


def plot_bernoulli_distribution(probability=0.5):
    try:
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        x = np.array([0, 1])
        y = stats.bernoulli.pmf(x, probability)

        fig, ax = plt.subplots()

        ax.bar(x, y)
        ax.set_title("Bernoulli Distribution")
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Probability")

        fig.tight_layout()

        return _success("bernoulli_distribution_plot", fig)

    except Exception as error:
        return _failure("bernoulli_distribution_plot", str(error))


# ------------------------------------------------------------
# BINOMIAL DISTRIBUTION
# ------------------------------------------------------------

def binomial_pmf(values, trials, probability=0.5):
    try:
        values = np.array(values, dtype=int)

        if not isinstance(trials, int) or trials < 1:
            raise ValueError("Number of trials must be a positive integer.")

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        if np.any((values < 0) | (values > trials)):
            raise ValueError(
                "Binomial outcomes must be between 0 and the number of trials."
            )

        result = stats.binom.pmf(
            values,
            n=trials,
            p=probability
        )

        return _success("binomial_pmf", result)

    except Exception as error:
        return _failure("binomial_pmf", str(error))


def binomial_cdf(values, trials, probability=0.5):
    try:
        values = np.array(values, dtype=float)

        if not isinstance(trials, int) or trials < 1:
            raise ValueError("Number of trials must be a positive integer.")

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        result = stats.binom.cdf(
            values,
            n=trials,
            p=probability
        )

        return _success("binomial_cdf", result)

    except Exception as error:
        return _failure("binomial_cdf", str(error))


def binomial_probability(successes, trials, probability=0.5):
    try:
        if not isinstance(trials, int) or trials < 1:
            raise ValueError("Number of trials must be a positive integer.")

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        if successes < 0 or successes > trials:
            raise ValueError(
                "Number of successes must be between 0 and trials."
            )

        result = stats.binom.pmf(
            successes,
            n=trials,
            p=probability
        )

        return _success("binomial_probability", result)

    except Exception as error:
        return _failure("binomial_probability", str(error))


def binomial_mean(trials, probability=0.5):
    try:
        if not isinstance(trials, int) or trials < 1:
            raise ValueError("Number of trials must be a positive integer.")

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        result = trials * probability

        return _success("binomial_mean", result)

    except Exception as error:
        return _failure("binomial_mean", str(error))


def binomial_variance(trials, probability=0.5):
    try:
        if not isinstance(trials, int) or trials < 1:
            raise ValueError("Number of trials must be a positive integer.")

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        result = trials * probability * (1 - probability)

        return _success("binomial_variance", result)

    except Exception as error:
        return _failure("binomial_variance", str(error))


def plot_binomial_distribution(trials, probability=0.5):
    try:
        if not isinstance(trials, int) or trials < 1:
            raise ValueError("Number of trials must be a positive integer.")

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        x = np.arange(0, trials + 1)
        y = stats.binom.pmf(
            x,
            n=trials,
            p=probability
        )

        fig, ax = plt.subplots()

        ax.bar(x, y)
        ax.set_title("Binomial Distribution")
        ax.set_xlabel("Number of Successes")
        ax.set_ylabel("Probability")

        fig.tight_layout()

        return _success("binomial_distribution_plot", fig)

    except Exception as error:
        return _failure("binomial_distribution_plot", str(error))


# ------------------------------------------------------------
# GEOMETRIC DISTRIBUTION
# ------------------------------------------------------------

def geometric_pmf(values, probability=0.5):
    try:
        values = np.array(values, dtype=int)

        if not 0 < probability <= 1:
            raise ValueError("Probability must be greater than 0 and at most 1.")

        if np.any(values < 1):
            raise ValueError(
                "Geometric trial numbers must be at least 1."
            )

        result = stats.geom.pmf(values, probability)

        return _success("geometric_pmf", result)

    except Exception as error:
        return _failure("geometric_pmf", str(error))


def geometric_cdf(values, probability=0.5):
    try:
        values = np.array(values, dtype=float)

        if not 0 < probability <= 1:
            raise ValueError("Probability must be greater than 0 and at most 1.")

        if np.any(values < 0):
            raise ValueError(
                "Geometric values cannot be negative."
            )

        result = stats.geom.cdf(values, probability)

        return _success("geometric_cdf", result)

    except Exception as error:
        return _failure("geometric_cdf", str(error))


def geometric_probability(trials, probability=0.5):
    try:
        if not isinstance(trials, int) or trials < 1:
            raise ValueError("Trials must be a positive integer.")

        if not 0 < probability <= 1:
            raise ValueError("Probability must be greater than 0 and at most 1.")

        result = stats.geom.pmf(trials, probability)

        return _success("geometric_probability", result)

    except Exception as error:
        return _failure("geometric_probability", str(error))


def geometric_mean(probability=0.5):
    try:
        if not 0 < probability <= 1:
            raise ValueError("Probability must be greater than 0 and at most 1.")

        result = 1 / probability

        return _success("geometric_mean", result)

    except Exception as error:
        return _failure("geometric_mean", str(error))


def geometric_variance(probability=0.5):
    try:
        if not 0 < probability <= 1:
            raise ValueError("Probability must be greater than 0 and at most 1.")

        result = (1 - probability) / (probability ** 2)

        return _success("geometric_variance", result)

    except Exception as error:
        return _failure("geometric_variance", str(error))


def plot_geometric_distribution(probability=0.5, max_trials=10):
    try:
        if not 0 < probability <= 1:
            raise ValueError("Probability must be greater than 0 and at most 1.")

        if not isinstance(max_trials, int) or max_trials < 1:
            raise ValueError("max_trials must be a positive integer.")

        x = np.arange(1, max_trials + 1)
        y = stats.geom.pmf(x, probability)

        fig, ax = plt.subplots()

        ax.bar(x, y)
        ax.set_title("Geometric Distribution")
        ax.set_xlabel("Trial of First Success")
        ax.set_ylabel("Probability")

        fig.tight_layout()

        return _success("geometric_distribution_plot", fig)

    except Exception as error:
        return _failure("geometric_distribution_plot", str(error))


# ------------------------------------------------------------
# POISSON DISTRIBUTION
# ------------------------------------------------------------

def poisson_pmf(values, rate=1):
    try:
        values = np.array(values, dtype=int)

        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        if np.any(values < 0):
            raise ValueError(
                "Poisson event counts cannot be negative."
            )

        result = stats.poisson.pmf(values, rate)

        return _success("poisson_pmf", result)

    except Exception as error:
        return _failure("poisson_pmf", str(error))


def poisson_cdf(values, rate=1):
    try:
        values = np.array(values, dtype=float)

        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        if np.any(values < 0):
            raise ValueError(
                "Poisson event counts cannot be negative."
            )

        result = stats.poisson.cdf(values, rate)

        return _success("poisson_cdf", result)

    except Exception as error:
        return _failure("poisson_cdf", str(error))


def poisson_probability(events, rate=1):
    try:
        if not isinstance(events, int) or events < 0:
            raise ValueError(
                "Number of events must be a non-negative integer."
            )

        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        result = stats.poisson.pmf(events, rate)

        return _success("poisson_probability", result)

    except Exception as error:
        return _failure("poisson_probability", str(error))


def poisson_mean(rate=1):
    try:
        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        return _success("poisson_mean", rate)

    except Exception as error:
        return _failure("poisson_mean", str(error))


def poisson_variance(rate=1):
    try:
        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        return _success("poisson_variance", rate)

    except Exception as error:
        return _failure("poisson_variance", str(error))


def plot_poisson_distribution(rate=1, max_events=15):
    try:
        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        if not isinstance(max_events, int) or max_events < 1:
            raise ValueError("max_events must be a positive integer.")

        x = np.arange(0, max_events + 1)
        y = stats.poisson.pmf(x, rate)

        fig, ax = plt.subplots()

        ax.bar(x, y)
        ax.set_title("Poisson Distribution")
        ax.set_xlabel("Number of Events")
        ax.set_ylabel("Probability")

        fig.tight_layout()

        return _success("poisson_distribution_plot", fig)

    except Exception as error:
        return _failure("poisson_distribution_plot", str(error))

# ============================================================
# L5.8 — EXPONENTIAL + NORMAL DISTRIBUTIONS
# ============================================================

# -------------------------
# EXPONENTIAL DISTRIBUTION
# -------------------------

def exponential_pdf(values, rate=1):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        if np.any(values < 0):
            raise ValueError("Exponential values cannot be negative.")

        result = stats.expon.pdf(values, scale=1 / rate)

        return _success("exponential_pdf", result)

    except Exception as error:
        return _failure("exponential_pdf", str(error))


def exponential_cdf(values, rate=1):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        if np.any(values < 0):
            raise ValueError("Exponential values cannot be negative.")

        result = stats.expon.cdf(values, scale=1 / rate)

        return _success("exponential_cdf", result)

    except Exception as error:
        return _failure("exponential_cdf", str(error))


def exponential_probability(lower_bound, upper_bound, rate=1):
    try:
        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        if lower_bound < 0 or upper_bound < 0:
            raise ValueError("Bounds cannot be negative.")

        if lower_bound > upper_bound:
            raise ValueError("Lower bound cannot exceed upper bound.")

        result = (
            stats.expon.cdf(upper_bound, scale=1 / rate)
            - stats.expon.cdf(lower_bound, scale=1 / rate)
        )

        return _success("exponential_probability", result)

    except Exception as error:
        return _failure("exponential_probability", str(error))


def exponential_mean(rate=1):
    try:
        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        result = 1 / rate

        return _success("exponential_mean", result)

    except Exception as error:
        return _failure("exponential_mean", str(error))


def exponential_variance(rate=1):
    try:
        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        result = 1 / (rate ** 2)

        return _success("exponential_variance", result)

    except Exception as error:
        return _failure("exponential_variance", str(error))


def sample_exponential(size=10, rate=1):
    try:
        if not isinstance(size, int) or size < 1:
            raise ValueError("Sample size must be a positive integer.")

        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        result = np.random.exponential(
            scale=1 / rate,
            size=size
        )

        return _success("sample_exponential", result)

    except Exception as error:
        return _failure("sample_exponential", str(error))


def plot_exponential_distribution(rate=1, maximum=10, points=200):
    try:
        if rate <= 0:
            raise ValueError("Rate must be greater than zero.")

        if maximum <= 0:
            raise ValueError("Maximum value must be greater than zero.")

        if not isinstance(points, int) or points < 2:
            raise ValueError("Points must be at least 2.")

        x = np.linspace(0, maximum, points)

        y = stats.expon.pdf(
            x,
            scale=1 / rate
        )

        fig, ax = plt.subplots()

        ax.plot(x, y)

        ax.set_title("Exponential Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

        fig.tight_layout()

        return _success("exponential_distribution_plot", fig)

    except Exception as error:
        return _failure("exponential_distribution_plot", str(error))


# -------------------------
# NORMAL DISTRIBUTION
# -------------------------

def normal_pdf(values, mean=0, standard_deviation=1):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if standard_deviation <= 0:
            raise ValueError("Standard deviation must be greater than zero.")

        result = stats.norm.pdf(
            values,
            loc=mean,
            scale=standard_deviation
        )

        return _success("normal_pdf", result)

    except Exception as error:
        return _failure("normal_pdf", str(error))


def normal_cdf(values, mean=0, standard_deviation=1):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if standard_deviation <= 0:
            raise ValueError("Standard deviation must be greater than zero.")

        result = stats.norm.cdf(
            values,
            loc=mean,
            scale=standard_deviation
        )

        return _success("normal_cdf", result)

    except Exception as error:
        return _failure("normal_cdf", str(error))


def normal_probability(lower_bound, upper_bound, mean=0, standard_deviation=1):
    try:
        if standard_deviation <= 0:
            raise ValueError("Standard deviation must be greater than zero.")

        if lower_bound > upper_bound:
            raise ValueError("Lower bound cannot exceed upper bound.")

        result = (
            stats.norm.cdf(
                upper_bound,
                loc=mean,
                scale=standard_deviation
            )
            -
            stats.norm.cdf(
                lower_bound,
                loc=mean,
                scale=standard_deviation
            )
        )

        return _success("normal_probability", result)

    except Exception as error:
        return _failure("normal_probability", str(error))


def normal_mean(mean=0):
    try:
        return _success("normal_mean", mean)

    except Exception as error:
        return _failure("normal_mean", str(error))


def normal_variance(standard_deviation=1):
    try:
        if standard_deviation <= 0:
            raise ValueError("Standard deviation must be greater than zero.")

        result = standard_deviation ** 2

        return _success("normal_variance", result)

    except Exception as error:
        return _failure("normal_variance", str(error))


def sample_normal(size=10, mean=0, standard_deviation=1):
    try:
        if not isinstance(size, int) or size < 1:
            raise ValueError("Sample size must be a positive integer.")

        if standard_deviation <= 0:
            raise ValueError("Standard deviation must be greater than zero.")

        result = np.random.normal(
            loc=mean,
            scale=standard_deviation,
            size=size
        )

        return _success("sample_normal", result)

    except Exception as error:
        return _failure("sample_normal", str(error))


def plot_normal_distribution(mean=0, standard_deviation=1, points=200):
    try:
        if standard_deviation <= 0:
            raise ValueError("Standard deviation must be greater than zero.")

        if not isinstance(points, int) or points < 2:
            raise ValueError("Points must be at least 2.")

        lower = mean - 4 * standard_deviation
        upper = mean + 4 * standard_deviation

        x = np.linspace(lower, upper, points)

        y = stats.norm.pdf(
            x,
            loc=mean,
            scale=standard_deviation
        )

        fig, ax = plt.subplots()

        ax.plot(x, y)

        ax.set_title("Normal Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

        fig.tight_layout()

        return _success("normal_distribution_plot", fig)

    except Exception as error:
        return _failure("normal_distribution_plot", str(error))


# -------------------------
# CHI-SQUARE DISTRIBUTION
# -------------------------

def chi_square_pdf(values, degrees_of_freedom):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if np.any(values < 0):
            raise ValueError("Chi-square values cannot be negative.")

        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        result = stats.chi2.pdf(
            values,
            df=degrees_of_freedom
        )

        return _success("chi_square_pdf", result)

    except Exception as error:
        return _failure("chi_square_pdf", str(error))


def chi_square_cdf(values, degrees_of_freedom):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if np.any(values < 0):
            raise ValueError("Chi-square values cannot be negative.")

        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        result = stats.chi2.cdf(
            values,
            df=degrees_of_freedom
        )

        return _success("chi_square_cdf", result)

    except Exception as error:
        return _failure("chi_square_cdf", str(error))


def chi_square_probability(
    lower_bound,
    upper_bound,
    degrees_of_freedom
):
    try:
        if lower_bound < 0 or upper_bound < 0:
            raise ValueError("Chi-square bounds cannot be negative.")

        if lower_bound > upper_bound:
            raise ValueError("Lower bound cannot exceed upper bound.")

        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        result = (
            stats.chi2.cdf(
                upper_bound,
                df=degrees_of_freedom
            )
            -
            stats.chi2.cdf(
                lower_bound,
                df=degrees_of_freedom
            )
        )

        return _success("chi_square_probability", result)

    except Exception as error:
        return _failure("chi_square_probability", str(error))


def chi_square_mean(degrees_of_freedom):
    try:
        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        return _success(
            "chi_square_mean",
            degrees_of_freedom
        )

    except Exception as error:
        return _failure("chi_square_mean", str(error))


def chi_square_variance(degrees_of_freedom):
    try:
        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        result = 2 * degrees_of_freedom

        return _success(
            "chi_square_variance",
            result
        )

    except Exception as error:
        return _failure("chi_square_variance", str(error))


def chi_square_quantile(probability, degrees_of_freedom):
    try:
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        result = stats.chi2.ppf(
            probability,
            df=degrees_of_freedom
        )

        return _success(
            "chi_square_quantile",
            result
        )

    except Exception as error:
        return _failure("chi_square_quantile", str(error))


def sample_chi_square(size=10, degrees_of_freedom=5):
    try:
        if not isinstance(size, int) or size < 1:
            raise ValueError("Sample size must be a positive integer.")

        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        result = np.random.chisquare(
            degrees_of_freedom,
            size=size
        )

        return _success(
            "sample_chi_square",
            result
        )

    except Exception as error:
        return _failure("sample_chi_square", str(error))


def plot_chi_square_distribution(
    degrees_of_freedom=5,
    points=300,
    maximum=20
):
    try:
        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        if points < 2:
            raise ValueError("Points must be at least 2.")

        if maximum <= 0:
            raise ValueError("Maximum must be greater than zero.")

        x = np.linspace(0, maximum, points)

        y = stats.chi2.pdf(
            x,
            df=degrees_of_freedom
        )

        fig, ax = plt.subplots()

        ax.plot(x, y)

        ax.set_title(
            f"Chi-Square Distribution (df={degrees_of_freedom})"
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

        fig.tight_layout()

        return _success(
            "chi_square_distribution_plot",
            fig
        )

    except Exception as error:
        return _failure(
            "chi_square_distribution_plot",
            str(error)
        )

# ============================================================
# L5.10A — T-DISTRIBUTION
# ============================================================

def t_pdf(values, degrees_of_freedom):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        result = stats.t.pdf(values, df=degrees_of_freedom)

        return _success("t_pdf", result)

    except Exception as error:
        return _failure("t_pdf", str(error))


def t_cdf(values, degrees_of_freedom):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Values cannot be empty.")

        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        result = stats.t.cdf(values, df=degrees_of_freedom)

        return _success("t_cdf", result)

    except Exception as error:
        return _failure("t_cdf", str(error))


def t_probability(lower_bound, upper_bound, degrees_of_freedom):
    try:
        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        if lower_bound > upper_bound:
            raise ValueError("Lower bound cannot exceed upper bound.")

        result = (
            stats.t.cdf(upper_bound, df=degrees_of_freedom)
            - stats.t.cdf(lower_bound, df=degrees_of_freedom)
        )

        return _success("t_probability", result)

    except Exception as error:
        return _failure("t_probability", str(error))


def t_mean(degrees_of_freedom):
    try:
        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        if degrees_of_freedom <= 1:
            return _success("t_mean", np.nan)

        return _success("t_mean", 0.0)

    except Exception as error:
        return _failure("t_mean", str(error))


def t_variance(degrees_of_freedom):
    try:
        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        if degrees_of_freedom <= 2:
            return _success("t_variance", np.inf)

        result = degrees_of_freedom / (degrees_of_freedom - 2)

        return _success("t_variance", result)

    except Exception as error:
        return _failure("t_variance", str(error))


def t_quantile(probability, degrees_of_freedom):
    try:
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        result = stats.t.ppf(
            probability,
            df=degrees_of_freedom
        )

        return _success("t_quantile", result)

    except Exception as error:
        return _failure("t_quantile", str(error))


def sample_t(size=10, degrees_of_freedom=10):
    try:
        if not isinstance(size, int) or size < 1:
            raise ValueError("Sample size must be a positive integer.")

        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        result = np.random.standard_t(
            degrees_of_freedom,
            size=size
        )

        return _success("sample_t", result)

    except Exception as error:
        return _failure("sample_t", str(error))


def plot_t_distribution(
    degrees_of_freedom=10,
    points=300,
    lower=-5,
    upper=5
):
    try:
        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be a positive integer.")

        if points < 2:
            raise ValueError("Points must be at least 2.")

        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound.")

        x = np.linspace(lower, upper, points)

        y = stats.t.pdf(
            x,
            df=degrees_of_freedom
        )

        fig, ax = plt.subplots()

        ax.plot(x, y)

        ax.set_title(
            f"t-Distribution (df={degrees_of_freedom})"
        )
        ax.set_xlabel("t")
        ax.set_ylabel("Density")

        fig.tight_layout()

        return _success("t_distribution_plot", fig)

    except Exception as error:
        return _failure("t_distribution_plot", str(error))

# ============================================================
# L5.11 — SAMPLING & SAMPLING DISTRIBUTIONS
# ============================================================

def random_sample(values, sample_size):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Population cannot be empty.")

        if not isinstance(sample_size, int) or sample_size < 1:
            raise ValueError("Sample size must be a positive integer.")

        if sample_size > values.size:
            raise ValueError("Sample size cannot exceed population size.")

        result = np.random.choice(
            values,
            size=sample_size,
            replace=False
        )

        return _success("random_sample", result)

    except Exception as error:
        return _failure("random_sample", str(error))


def sample_mean(values):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Sample cannot be empty.")

        return _success("sample_mean", np.mean(values))

    except Exception as error:
        return _failure("sample_mean", str(error))


def sample_variance(values):
    try:
        values = np.array(values, dtype=float)

        if values.size < 2:
            raise ValueError("At least two values are required.")

        return _success(
            "sample_variance",
            np.var(values, ddof=1)
        )

    except Exception as error:
        return _failure("sample_variance", str(error))


def standard_error(values):
    try:
        values = np.array(values, dtype=float)

        if values.size < 2:
            raise ValueError("At least two values are required.")

        result = np.std(values, ddof=1) / np.sqrt(values.size)

        return _success("standard_error", result)

    except Exception as error:
        return _failure("standard_error", str(error))


def sampling_distribution_means(
    values,
    sample_size,
    number_of_samples=1000
):
    try:
        values = np.array(values, dtype=float)

        if values.size == 0:
            raise ValueError("Population cannot be empty.")

        if sample_size < 1 or sample_size > values.size:
            raise ValueError(
                "Sample size must be between 1 and population size."
            )

        if not isinstance(number_of_samples, int) or number_of_samples < 1:
            raise ValueError(
                "Number of samples must be a positive integer."
            )

        means = []

        for _ in range(number_of_samples):
            sample = np.random.choice(
                values,
                size=sample_size,
                replace=True
            )
            means.append(np.mean(sample))

        return _success(
            "sampling_distribution_means",
            np.array(means)
        )

    except Exception as error:
        return _failure(
            "sampling_distribution_means",
            str(error)
        )


def plot_sampling_distribution(
    values,
    sample_size,
    number_of_samples=1000
):
    try:
        result = sampling_distribution_means(
            values,
            sample_size,
            number_of_samples
        )

        if not result["success"]:
            return result

        means = result["result"]

        fig, ax = plt.subplots()

        sns.histplot(
            means,
            kde=True,
            ax=ax
        )

        ax.set_title("Sampling Distribution of Sample Means")
        ax.set_xlabel("Sample Mean")
        ax.set_ylabel("Frequency")

        fig.tight_layout()

        return _success(
            "sampling_distribution_plot",
            fig
        )

    except Exception as error:
        return _failure(
            "sampling_distribution_plot",
            str(error)
        )

# ============================================================
# L5.12 — CENTRAL LIMIT THEOREM
# ============================================================

def central_limit_theorem(
    population,
    sample_size,
    number_of_samples=1000
):
    try:
        population = np.array(population, dtype=float)

        if population.size == 0:
            raise ValueError("Population cannot be empty.")

        if sample_size < 1:
            raise ValueError("Sample size must be positive.")

        if number_of_samples < 1:
            raise ValueError(
                "Number of samples must be positive."
            )

        sample_means = []

        for _ in range(number_of_samples):
            sample = np.random.choice(
                population,
                size=sample_size,
                replace=True
            )

            sample_means.append(np.mean(sample))

        sample_means = np.array(sample_means)

        return _success(
            "central_limit_theorem",
            sample_means
        )

    except Exception as error:
        return _failure(
            "central_limit_theorem",
            str(error)
        )


def clt_standard_error(population, sample_size):
    try:
        population = np.array(population, dtype=float)

        if population.size < 2:
            raise ValueError("Population needs at least two values.")

        if sample_size < 1:
            raise ValueError("Sample size must be positive.")

        result = np.std(population, ddof=0) / np.sqrt(sample_size)

        return _success(
            "clt_standard_error",
            result
        )

    except Exception as error:
        return _failure(
            "clt_standard_error",
            str(error)
        )


def plot_clt(
    population,
    sample_size,
    number_of_samples=1000
):
    try:
        result = central_limit_theorem(
            population,
            sample_size,
            number_of_samples
        )

        if not result["success"]:
            return result

        means = result["result"]

        fig, ax = plt.subplots()

        sns.histplot(
            means,
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"Central Limit Theorem (n={sample_size})"
        )
        ax.set_xlabel("Sample Mean")
        ax.set_ylabel("Frequency")

        fig.tight_layout()

        return _success(
            "clt_plot",
            fig
        )

    except Exception as error:
        return _failure(
            "clt_plot",
            str(error)
        )


# ============================================================
# L5.13 — LAW OF LARGE NUMBERS
# ============================================================

def law_of_large_numbers(
    population,
    number_of_samples=1000
):
    try:
        population = np.array(population, dtype=float)

        if population.size == 0:
            raise ValueError("Population cannot be empty.")

        if number_of_samples < 1:
            raise ValueError(
                "Number of samples must be positive."
            )

        samples = np.random.choice(
            population,
            size=number_of_samples,
            replace=True
        )

        cumulative_mean = np.cumsum(samples) / np.arange(
            1,
            number_of_samples + 1
        )

        return _success(
            "law_of_large_numbers",
            cumulative_mean
        )

    except Exception as error:
        return _failure(
            "law_of_large_numbers",
            str(error)
        )


def plot_law_of_large_numbers(
    population,
    number_of_samples=1000
):
    try:
        result = law_of_large_numbers(
            population,
            number_of_samples
        )

        if not result["success"]:
            return result

        cumulative_mean = result["result"]
        expected_value = np.mean(population)

        fig, ax = plt.subplots()

        ax.plot(
            cumulative_mean,
            label="Cumulative Mean"
        )

        ax.axhline(
            expected_value,
            linestyle="--",
            label="Expected Value"
        )

        ax.set_title("Law of Large Numbers")
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel("Mean")
        ax.legend()

        fig.tight_layout()

        return _success(
            "lln_plot",
            fig
        )

    except Exception as error:
        return _failure(
            "lln_plot",
            str(error)
        )

# ============================================================
# L5.14 — STATISTICAL TESTS
# ============================================================

def one_sample_t_test(values, population_mean):
    try:
        values = np.array(values, dtype=float)

        if values.size < 2:
            raise ValueError(
                "At least two observations are required."
            )

        statistic, p_value = stats.ttest_1samp(
            values,
            population_mean
        )

        return _success(
            "one_sample_t_test",
            {
                "statistic": statistic,
                "p_value": p_value
            }
        )

    except Exception as error:
        return _failure(
            "one_sample_t_test",
            str(error)
        )


def independent_t_test(values_a, values_b):
    try:
        values_a = np.array(values_a, dtype=float)
        values_b = np.array(values_b, dtype=float)

        if values_a.size < 2 or values_b.size < 2:
            raise ValueError(
                "Both groups require at least two observations."
            )

        statistic, p_value = stats.ttest_ind(
            values_a,
            values_b
        )

        return _success(
            "independent_t_test",
            {
                "statistic": statistic,
                "p_value": p_value
            }
        )

    except Exception as error:
        return _failure(
            "independent_t_test",
            str(error)
        )


def paired_t_test(values_a, values_b):
    try:
        values_a = np.array(values_a, dtype=float)
        values_b = np.array(values_b, dtype=float)

        if values_a.size != values_b.size:
            raise ValueError(
                "Paired samples must have the same size."
            )

        if values_a.size < 2:
            raise ValueError(
                "At least two paired observations are required."
            )

        statistic, p_value = stats.ttest_rel(
            values_a,
            values_b
        )

        return _success(
            "paired_t_test",
            {
                "statistic": statistic,
                "p_value": p_value
            }
        )

    except Exception as error:
        return _failure(
            "paired_t_test",
            str(error)
        )


def chi_square_goodness_of_fit(
    observed,
    expected=None
):
    try:
        observed = np.array(observed, dtype=float)

        if observed.size == 0:
            raise ValueError(
                "Observed values cannot be empty."
            )

        if np.any(observed < 0):
            raise ValueError(
                "Observed frequencies cannot be negative."
            )

        if expected is None:
            expected = np.full(
                observed.size,
                np.sum(observed) / observed.size
            )
        else:
            expected = np.array(
                expected,
                dtype=float
            )

        if expected.size != observed.size:
            raise ValueError(
                "Observed and expected values must have the same size."
            )

        if np.any(expected <= 0):
            raise ValueError(
                "Expected frequencies must be greater than zero."
            )

        statistic, p_value = stats.chisquare(
            observed,
            f_exp=expected
        )

        return _success(
            "chi_square_goodness_of_fit",
            {
                "statistic": statistic,
                "p_value": p_value
            }
        )

    except Exception as error:
        return _failure(
            "chi_square_goodness_of_fit",
            str(error)
        )


def chi_square_independence(contingency_table):
    try:
        table = np.array(
            contingency_table,
            dtype=float
        )

        if table.ndim != 2:
            raise ValueError(
                "Contingency table must be two-dimensional."
            )

        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError(
                "Contingency table must contain at least 2 rows and 2 columns."
            )

        if np.any(table < 0):
            raise ValueError(
                "Contingency table cannot contain negative values."
            )

        statistic, p_value, degrees_of_freedom, expected = (
            stats.chi2_contingency(table)
        )

        return _success(
            "chi_square_independence",
            {
                "statistic": statistic,
                "p_value": p_value,
                "degrees_of_freedom": degrees_of_freedom,
                "expected": expected
            }
        )

    except Exception as error:
        return _failure(
            "chi_square_independence",
            str(error)
        )


def interpret_p_value(p_value, alpha=0.05):
    try:
        if not 0 <= p_value <= 1:
            raise ValueError(
                "p-value must be between 0 and 1."
            )

        if not 0 < alpha < 1:
            raise ValueError(
                "Significance level must be between 0 and 1."
            )

        if p_value < alpha:
            decision = "Reject the null hypothesis."
        else:
            decision = "Fail to reject the null hypothesis."

        return _success(
            "p_value_interpretation",
            {
                "p_value": p_value,
                "alpha": alpha,
                "decision": decision
            }
        )

    except Exception as error:
        return _failure(
            "p_value_interpretation",
            str(error)
        )

