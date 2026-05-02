from __future__ import annotations

import ast
import re
from typing import Callable

import numpy as np
from scipy.integrate import quad


ALLOWED_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
}

ALLOWED_CONSTANTS = {
    "pi": np.pi,
    "e": np.e,
}


class ExpressionValidationError(ValueError):
    """Raised when expression uses unsupported syntax."""


def _validate_ast(node: ast.AST) -> None:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
    )

    for child in ast.walk(node):
        if not isinstance(child, allowed_nodes):
            raise ExpressionValidationError(f"Unsupported syntax: {type(child).__name__}")
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name):
                raise ExpressionValidationError("Only direct function calls are allowed.")
            if child.func.id not in ALLOWED_FUNCTIONS:
                raise ExpressionValidationError(f"Function '{child.func.id}' is not allowed.")
        if isinstance(child, ast.Name):
            if child.id not in ALLOWED_FUNCTIONS and child.id not in ALLOWED_CONSTANTS and child.id != "x":
                raise ExpressionValidationError(f"Name '{child.id}' is not allowed.")


def compile_function(expression: str) -> Callable[[np.ndarray], np.ndarray]:
    expression = expression.strip()
    if not expression:
        raise ExpressionValidationError("Function expression cannot be empty.")

    parsed = ast.parse(expression, mode="eval")
    _validate_ast(parsed)
    compiled = compile(parsed, "<expression>", "eval")

    scope = {"__builtins__": {}}
    scope.update(ALLOWED_FUNCTIONS)
    scope.update(ALLOWED_CONSTANTS)

    def func(x: np.ndarray) -> np.ndarray:
        local_scope = {"x": x}
        return np.asarray(eval(compiled, scope, local_scope), dtype=float)

    return func


def numerical_derivative(func: Callable[[np.ndarray], np.ndarray], x_values: np.ndarray, h: float = 1e-5) -> np.ndarray:
    return (func(x_values + h) - func(x_values - h)) / (2.0 * h)


def nth_numerical_derivative(
    func: Callable[[np.ndarray], np.ndarray],
    x_values: np.ndarray,
    order: int,
    h: float = 1e-5,
) -> np.ndarray:
    if order < 0:
        raise ValueError("Derivative order must be non-negative.")
    if order == 0:
        return func(x_values)

    derivative_func = func
    for _ in range(order):
        previous = derivative_func

        def derivative_func(x: np.ndarray, prev=previous) -> np.ndarray:
            return (prev(x + h) - prev(x - h)) / (2.0 * h)

    return derivative_func(x_values)


def numerical_integration(func: Callable[[np.ndarray], np.ndarray], a: float, b: float) -> float:
    result, _ = quad(lambda t: float(func(np.array([t]))[0]), a, b)
    return float(result)


def area_between_curves(
    func1: Callable[[np.ndarray], np.ndarray],
    func2: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
) -> float:
    diff = lambda t: abs(float(func1(np.array([t]))[0]) - float(func2(np.array([t]))[0]))
    result, _ = quad(diff, a, b)
    return float(result)


def polynomial_derivative_expression(expression: str) -> str | None:
    """Return symbolic derivative for simple polynomial input, else None."""
    compact = expression.replace(" ", "")
    if not compact:
        return None
    if compact[0] not in "+-":
        compact = f"+{compact}"

    term_matches = re.findall(r"[+-][^+-]+", compact)
    if not term_matches:
        return None

    derivative_terms: list[tuple[int, float]] = []
    for term in term_matches:
        try:
            sign = -1.0 if term[0] == "-" else 1.0
            body = term[1:]

            if "x" not in body:
                continue

            coeff = 1.0
            power = 1
            if body == "x":
                coeff = 1.0
                power = 1
            elif body.endswith("*x"):
                coeff_part = body[:-2]
                coeff = float(coeff_part)
                power = 1
            elif "x**" in body:
                if body.startswith("x**"):
                    coeff = 1.0
                    power = int(body[3:])
                elif "*x**" in body:
                    coeff_part, power_part = body.split("*x**")
                    coeff = float(coeff_part)
                    power = int(power_part)
                else:
                    return None
            else:
                return None
        except (ValueError, TypeError):
            return None

        coeff *= sign
        if power == 0:
            continue
        derivative_terms.append((power - 1, coeff * power))

    if not derivative_terms:
        return "0"

    derivative_terms.sort(key=lambda item: item[0], reverse=True)
    pieces: list[str] = []
    for idx, (power, coeff) in enumerate(derivative_terms):
        if abs(coeff) < 1e-12:
            continue

        sign = "-" if coeff < 0 else "+"
        abs_coeff = abs(coeff)
        coeff_str = str(int(abs_coeff)) if abs_coeff.is_integer() else f"{abs_coeff:g}"

        if power == 0:
            body = coeff_str
        elif power == 1:
            body = "x" if coeff_str == "1" else f"{coeff_str}*x"
        else:
            body = f"x**{power}" if coeff_str == "1" else f"{coeff_str}*x**{power}"

        if idx == 0:
            pieces.append(body if sign == "+" else f"-{body}")
        else:
            pieces.append(f" {sign} {body}")

    return "".join(pieces) if pieces else "0"

