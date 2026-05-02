from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .math_engine import (
    ExpressionValidationError,
    area_between_curves,
    compile_function,
    nth_numerical_derivative,
    numerical_integration,
    polynomial_derivative_expression,
)


class CalctoolApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Calculus Tool App")
        self.root.geometry("1280x800")

        self._build_ui()

    def _build_ui(self) -> None:
        self.main_frame = ttk.Frame(self.root, padding=12)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(self.main_frame, text="Inputs", padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.function_var = tk.StringVar(value="x**2 + 3*x + 5")
        self.second_function_var = tk.StringVar(value="2*x + 1")
        self.x_min_var = tk.StringVar(value="-10")
        self.x_max_var = tk.StringVar(value="10")
        self.points_var = tk.StringVar(value="500")
        self.integral_a_var = tk.StringVar(value="0")
        self.integral_b_var = tk.StringVar(value="5")
        self.between_curves_var = tk.BooleanVar(value=False)
        self.integral_result_var = tk.StringVar(value="Integral: -")
        self.derivative_summary_var = tk.StringVar(value="f'(x) samples: -")
        self.symbolic_derivative_var = tk.StringVar(value="Symbolic derivative (if polynomial): -")
        self.show_graph_var = tk.BooleanVar(value=True)
        self.derivative_mode_var = tk.StringVar(value="specific")
        self.derivative_order_var = tk.StringVar(value="1")
        self.auto_max_order_var = tk.StringVar(value="6")

        self._add_entry(controls, "f(x) =", self.function_var)
        self._add_entry(controls, "g(x) = (2nd curve)", self.second_function_var)
        ttk.Checkbutton(
            controls,
            text="Area between f and g on [a, b]",
            variable=self.between_curves_var,
        ).pack(anchor=tk.W, pady=(0, 4))
        self._add_entry(controls, "x min", self.x_min_var)
        self._add_entry(controls, "x max", self.x_max_var)
        self._add_entry(controls, "points", self.points_var)
        self._add_entry(controls, "integral a", self.integral_a_var)
        self._add_entry(controls, "integral b", self.integral_b_var)

        ttk.Separator(controls).pack(fill=tk.X, pady=8)
        ttk.Checkbutton(controls, text="Show original graph f(x)", variable=self.show_graph_var).pack(anchor=tk.W)
        ttk.Label(controls, text="Derivative mode").pack(anchor=tk.W, pady=(6, 0))
        ttk.Radiobutton(controls, text="No derivative output", variable=self.derivative_mode_var, value="none").pack(anchor=tk.W)
        ttk.Radiobutton(controls, text="Specific derivative order", variable=self.derivative_mode_var, value="specific").pack(anchor=tk.W)
        ttk.Radiobutton(controls, text="Auto derive until flat/limit", variable=self.derivative_mode_var, value="auto").pack(anchor=tk.W)
        self._add_entry(controls, "specific order n", self.derivative_order_var)
        self._add_entry(controls, "auto max order", self.auto_max_order_var)

        ttk.Button(controls, text="Plot Function", command=self.plot).pack(fill=tk.X, pady=(8, 4))
        ttk.Button(controls, text="Save Graph", command=self.save_graph).pack(fill=tk.X, pady=4)
        ttk.Button(controls, text="Clear", command=self.clear_plot).pack(fill=tk.X, pady=4)

        ttk.Separator(controls).pack(fill=tk.X, pady=10)
        ttk.Label(controls, textvariable=self.integral_result_var, foreground="darkgreen").pack(anchor=tk.W)
        ttk.Label(controls, textvariable=self.derivative_summary_var, foreground="navy").pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(controls, textvariable=self.symbolic_derivative_var, foreground="purple").pack(anchor=tk.W, pady=(4, 0))

        viz_frame = ttk.Frame(self.main_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        fig_frame = ttk.Frame(viz_frame)
        fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_entry(self, parent: ttk.Widget, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        ttk.Entry(parent, textvariable=variable, width=24).pack(fill=tk.X, pady=(0, 6))

    def plot(self) -> None:
        try:
            func = compile_function(self.function_var.get())
            x_min = float(self.x_min_var.get())
            x_max = float(self.x_max_var.get())
            points = int(self.points_var.get())
            a = float(self.integral_a_var.get())
            b = float(self.integral_b_var.get())

            if x_max <= x_min:
                raise ValueError("x max must be greater than x min.")
            if points < 50:
                raise ValueError("points must be at least 50 for smooth plots.")

            x_values = np.linspace(x_min, x_max, points)
            y_values = func(x_values)
            between_mode = self.between_curves_var.get()
            g_func = None
            y_g = None
            if between_mode:
                g_func = compile_function(self.second_function_var.get())
                y_g = g_func(x_values)
            derivative_mode = self.derivative_mode_var.get()

            self.axes.clear()
            if self.show_graph_var.get():
                self.axes.plot(x_values, y_values, label="f(x)", color="royalblue")
                if between_mode and y_g is not None:
                    self.axes.plot(x_values, y_g, label="g(x)", color="darkorange")

            derivative_label = ""
            if derivative_mode == "specific":
                order = int(self.derivative_order_var.get())
                if order < 1:
                    raise ValueError("specific order must be at least 1.")
                derivative_values = nth_numerical_derivative(func, x_values, order)
                derivative_label = f"f^({order})(x)"
                self.axes.plot(x_values, derivative_values, label=derivative_label, color="tomato")
                self.derivative_summary_var.set(f"{derivative_label} at mid x = {derivative_values[len(x_values)//2]:.6f}")
            elif derivative_mode == "auto":
                max_order = int(self.auto_max_order_var.get())
                if max_order < 1:
                    raise ValueError("auto max order must be at least 1.")
                final_order = 0
                latest_values = y_values
                for current_order in range(1, max_order + 1):
                    latest_values = nth_numerical_derivative(func, x_values, current_order)
                    final_order = current_order
                    if np.allclose(latest_values, 0.0, atol=1e-4, rtol=0.0):
                        break
                derivative_label = f"f^({final_order})(x)"
                self.axes.plot(x_values, latest_values, label=derivative_label, color="tomato")
                self.derivative_summary_var.set(
                    f"Auto mode stopped at order {final_order}; {derivative_label} mid x = {latest_values[len(x_values)//2]:.6f}"
                )
            else:
                self.derivative_summary_var.set("Derivative mode: none")

            if self.show_graph_var.get():
                mask = (x_values >= min(a, b)) & (x_values <= max(a, b))
                if between_mode and y_g is not None:
                    y_lo = np.minimum(y_values[mask], y_g[mask])
                    y_hi = np.maximum(y_values[mask], y_g[mask])
                    self.axes.fill_between(
                        x_values[mask],
                        y_lo,
                        y_hi,
                        alpha=0.3,
                        color="mediumpurple",
                        label="Area between f and g",
                    )
                else:
                    self.axes.fill_between(
                        x_values[mask],
                        0,
                        y_values[mask],
                        alpha=0.25,
                        color="seagreen",
                        label="Integral Area (under f)",
                    )

            self.axes.axhline(0, color="black", linewidth=0.8)
            self.axes.grid(True, linestyle="--", alpha=0.4)
            handles, labels = self.axes.get_legend_handles_labels()
            if handles:
                self.axes.legend(loc="best")
            self.axes.set_title("Calculus-Powered Graphing App")
            self.axes.set_xlabel("x")
            self.axes.set_ylabel("y")
            self.canvas.draw()

            integral_result = numerical_integration(func, a, b)
            if between_mode and g_func is not None:
                between_area = area_between_curves(func, g_func, a, b)
                self.integral_result_var.set(
                    f"∫ f dx [{a}, {b}] = {integral_result:.6f}  |  "
                    f"Area between f & g [{a}, {b}] = {between_area:.6f}"
                )
            else:
                self.integral_result_var.set(f"Integral of f [{a}, {b}] = {integral_result:.6f}")
            symbolic = polynomial_derivative_expression(self.function_var.get())
            if symbolic is None:
                self.symbolic_derivative_var.set("Symbolic derivative (if polynomial): N/A")
            else:
                self.symbolic_derivative_var.set(f"Symbolic derivative: f'(x) = {symbolic}")
        except ExpressionValidationError as exc:
            messagebox.showerror("Invalid Function", str(exc))
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))

    def save_graph(self) -> None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
        )
        if file_path:
            self.figure.savefig(file_path)
            messagebox.showinfo("Saved", f"Graph saved to:\n{file_path}")

    def clear_plot(self) -> None:
        self.axes.clear()
        self.axes.set_title("Graph Cleared")
        self.canvas.draw()
        self.integral_result_var.set("Integral: -")
        self.derivative_summary_var.set("f'(x) samples: -")
        self.symbolic_derivative_var.set("Symbolic derivative (if polynomial): -")


def run() -> None:
    root = tk.Tk()
    app = CalctoolApp(root)
    root.mainloop()

