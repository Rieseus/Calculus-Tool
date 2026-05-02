from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes

from .math_engine import numerical_derivative


@dataclass
class PlotData:
    x_values: np.ndarray
    y_values: np.ndarray
    derivative_values: np.ndarray


def build_plot_data(func, x_min: float, x_max: float, points: int) -> PlotData:
    x_values = np.linspace(x_min, x_max, points)
    y_values = func(x_values)
    derivative_values = numerical_derivative(func, x_values)
    return PlotData(x_values=x_values, y_values=y_values, derivative_values=derivative_values)


def draw_plots(ax: Axes, data: PlotData, integral_a: float, integral_b: float) -> None:
    ax.clear()
    ax.plot(data.x_values, data.y_values, label="f(x)", color="royalblue")
    ax.plot(data.x_values, data.derivative_values, label="f'(x)", color="tomato")

    mask = (data.x_values >= min(integral_a, integral_b)) & (data.x_values <= max(integral_a, integral_b))
    ax.fill_between(
        data.x_values[mask],
        0,
        data.y_values[mask],
        alpha=0.25,
        color="seagreen",
        label="Integral Area",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    ax.set_title("Calculus Tool App")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

