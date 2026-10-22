from __future__ import annotations

import math
import tkinter as tk

from ecit.engine import StepSnapshotCallback
from ecit.math import Matrix


class BoundaryPlotterWindow:
    
    def __init__(
        self,
        X: Matrix,
        T: Matrix,
        snapshot_cb: StepSnapshotCallback,
        title: str = "ECIT - 2D Decision Boundary Visualizer",
        world_bounds: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
        canvas_size: int = 600,
        output_index: int = 0,
    ) -> None:
        if X.shape()[1] != 2:
            raise ValueError(
                f"BoundaryPlotterWindow requires 2D inputs (shape (P, 2)), got {X.shape()}"
            )
        if X.shape()[0] != T.shape()[0]:
            raise ValueError("Number of samples in X and T must match")
        if len(snapshot_cb.records) == 0:
            raise ValueError(
                "StepSnapshotCallback has no recorded steps. Run Trainer.fit() first."
            )

        self.X = X
        self.T = T
        self.snapshot_cb = snapshot_cb
        self.title = title
        self.x_min, self.x_max, self.y_min, self.y_max = [
            float(v) for v in world_bounds
        ]
        self.canvas_size = int(canvas_size)
        self.pad = 45
        self.output_index = output_index
        self.current_idx = 0

        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        self._build_ui()
        self._render_current_step()

    def _build_ui(self) -> None:
        self.info_label = tk.Label(
            self.root,
            text="",
            font=("Consolas", 11, "bold"),
            bg="#1e1e2e",
            fg="#cdd6f4",
            pady=8,
        )
        self.info_label.pack(fill=tk.X)

        self.eq_label = tk.Label(
            self.root,
            text="",
            font=("Consolas", 10),
            bg="#1e1e2e",
            fg="#89b4fa",
            pady=2,
        )
        self.eq_label.pack(fill=tk.X)

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#11111b",
            highlightthickness=1,
            highlightbackground="#45475a",
        )
        self.canvas.pack(padx=12, pady=8)

        ctrl_frame = tk.Frame(self.root, bg="#1e1e2e", pady=8)
        ctrl_frame.pack(fill=tk.X, padx=12)

        btn_style = {
            "font": ("Consolas", 10, "bold"),
            "bg": "#313244",
            "fg": "#cdd6f4",
            "activebackground": "#45475a",
            "activeforeground": "#ffffff",
            "relief": tk.FLAT,
            "padx": 12,
            "pady": 4,
        }

        self.btn_first = tk.Button(
            ctrl_frame, text="|< First", command=self.first_step, **btn_style
        )
        self.btn_first.pack(side=tk.LEFT, padx=4)

        self.btn_prev = tk.Button(
            ctrl_frame, text="< Prev Step", command=self.prev_step, **btn_style
        )
        self.btn_prev.pack(side=tk.LEFT, padx=4)

        self.step_counter_label = tk.Label(
            ctrl_frame,
            text="",
            font=("Consolas", 10, "bold"),
            bg="#1e1e2e",
            fg="#a6adc8",
        )
        self.step_counter_label.pack(side=tk.LEFT, expand=True)

        self.btn_next = tk.Button(
            ctrl_frame, text="Next Step >", command=self.next_step, **btn_style
        )
        self.btn_next.pack(side=tk.RIGHT, padx=4)

        self.btn_last = tk.Button(
            ctrl_frame, text="Last >|", command=self.last_step, **btn_style
        )
        self.btn_last.pack(side=tk.RIGHT, padx=4)

        self.root.bind("<Left>", lambda _e: self.prev_step())
        self.root.bind("<Right>", lambda _e: self.next_step())
        self.root.bind("<Home>", lambda _e: self.first_step())
        self.root.bind("<End>", lambda _e: self.last_step())

    def _world_to_pixel(self, x1: float, x2: float) -> tuple[float, float]:
        plot_w = self.canvas_size - 2 * self.pad
        plot_h = self.canvas_size - 2 * self.pad
        u = self.pad + ((x1 - self.x_min) / (self.x_max - self.x_min)) * plot_w
        v = (
            self.canvas_size
            - self.pad
            - ((x2 - self.y_min) / (self.y_max - self.y_min)) * plot_h
        )
        return (u, v)

    def _compute_clipped_line(
        self, w1: float, w2: float, rhs: float, eps: float = 1e-9
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if abs(w1) <= eps and abs(w2) <= eps:
            return None

        candidates: list[tuple[float, float]] = []

        if abs(w2) > eps:
            for x_bound in (self.x_min, self.x_max):
                y_val = (rhs - w1 * x_bound) / w2
                if self.y_min - eps <= y_val <= self.y_max + eps:
                    candidates.append((x_bound, max(self.y_min, min(self.y_max, y_val))))

        if abs(w1) > eps:
            for y_bound in (self.y_min, self.y_max):
                x_val = (rhs - w2 * y_bound) / w1
                if self.x_min - eps <= x_val <= self.x_max + eps:
                    candidates.append((max(self.x_min, min(self.x_max, x_val)), y_bound))

        unique_pts: list[tuple[float, float]] = []
        for pt in candidates:
            if not any(
                math.hypot(pt[0] - u[0], pt[1] - u[1]) < 1e-5 for u in unique_pts
            ):
                unique_pts.append(pt)

        if len(unique_pts) >= 2:
            return (unique_pts[0], unique_pts[1])
        return None

    def _draw_grid_and_axes(self) -> None:
        for tick in range(math.ceil(self.x_min), math.floor(self.x_max) + 1):
            u, _ = self._world_to_pixel(float(tick), 0.0)
            _, v_top = self._world_to_pixel(0.0, self.y_max)
            _, v_bot = self._world_to_pixel(0.0, self.y_min)
            self.canvas.create_line(u, v_top, u, v_bot, fill="#1e1e2e", width=1)

        for tick in range(math.ceil(self.y_min), math.floor(self.y_max) + 1):
            _, v = self._world_to_pixel(0.0, float(tick))
            u_left, _ = self._world_to_pixel(self.x_min, 0.0)
            u_right, _ = self._world_to_pixel(self.x_max, 0.0)
            self.canvas.create_line(u_left, v, u_right, v, fill="#1e1e2e", width=1)

        u_min, v_origin = self._world_to_pixel(self.x_min, 0.0)
        u_max, _ = self._world_to_pixel(self.x_max, 0.0)
        u_origin, v_min = self._world_to_pixel(0.0, self.y_min)
        _, v_max = self._world_to_pixel(0.0, self.y_max)

        self.canvas.create_line(
            u_min, v_origin, u_max, v_origin, fill="#6c7086", width=2, arrow=tk.LAST
        )
        self.canvas.create_line(
            u_origin, v_min, u_origin, v_max, fill="#6c7086", width=2, arrow=tk.LAST
        )

        self.canvas.create_text(
            u_max - 12,
            v_origin - 14,
            text="x1",
            fill="#cdd6f4",
            font=("Consolas", 11, "bold"),
        )
        self.canvas.create_text(
            u_origin + 16,
            v_max + 10,
            text="x2",
            fill="#cdd6f4",
            font=("Consolas", 11, "bold"),
        )

        for tick in range(math.ceil(self.x_min), math.floor(self.x_max) + 1):
            if tick == 0:
                continue
            u, _ = self._world_to_pixel(float(tick), 0.0)
            self.canvas.create_line(
                u, v_origin - 4, u, v_origin + 4, fill="#a6adc8", width=1
            )
            self.canvas.create_text(
                u,
                v_origin + 14,
                text=str(tick),
                fill="#a6adc8",
                font=("Consolas", 8),
            )

        for tick in range(math.ceil(self.y_min), math.floor(self.y_max) + 1):
            if tick == 0:
                continue
            _, v = self._world_to_pixel(0.0, float(tick))
            self.canvas.create_line(
                u_origin - 4, v, u_origin + 4, v, fill="#a6adc8", width=1
            )
            self.canvas.create_text(
                u_origin - 14,
                v,
                text=str(tick),
                fill="#a6adc8",
                font=("Consolas", 8),
            )

    def _render_current_step(self) -> None:
        self.canvas.delete("all")
        self._draw_grid_and_axes()

        record = self.snapshot_cb.records[self.current_idx]
        w_col, b_val, theta = self.snapshot_cb.get_hyperplane_at(
            self.current_idx, self.output_index
        )
        w1 = w_col.matrix[0][0]
        w2 = w_col.matrix[1][0]

        if theta > 1e-9:
            boundaries = [
                (theta - b_val, "#f9e2af", (6, 4), f"+θ ({theta:+.2f})"),
                (-theta - b_val, "#fab387", (6, 4), f"-θ ({-theta:+.2f})"),
            ]
            center_seg = self._compute_clipped_line(w1, w2, -b_val)
            if center_seg is not None:
                p1 = self._world_to_pixel(*center_seg[0])
                p2 = self._world_to_pixel(*center_seg[1])
                self.canvas.create_line(
                    p1[0], p1[1], p2[0], p2[1], fill="#585b70", width=1, dash=(2, 4)
                )
        else:
            boundaries = [(-b_val, "#f9e2af", None, "Boundary (y_NI = 0)")]

        any_line_drawn = False
        for rhs, color, dash_pat, _label in boundaries:
            seg = self._compute_clipped_line(w1, w2, rhs)
            if seg is not None:
                any_line_drawn = True
                p1 = self._world_to_pixel(*seg[0])
                p2 = self._world_to_pixel(*seg[1])
                if dash_pat:
                    self.canvas.create_line(
                        p1[0], p1[1], p2[0], p2[1], fill=color, width=3, dash=dash_pat
                    )
                else:
                    self.canvas.create_line(
                        p1[0], p1[1], p2[0], p2[1], fill=color, width=3
                    )

        w_norm = math.hypot(w1, w2)
        if w_norm > 1e-9:
            max_arrow_len = 1.5
            scale = min(1.0, max_arrow_len / w_norm)
            u0, v0 = self._world_to_pixel(0.0, 0.0)
            u1, v1 = self._world_to_pixel(w1 * scale, w2 * scale)
            self.canvas.create_line(
                u0, v0, u1, v1, fill="#89b4fa", width=2, arrow=tk.LAST
            )
            self.canvas.create_text(
                u1 + 12,
                v1 - 10,
                text=f"W=[{w1:.1f},{w2:.1f}]",
                fill="#89b4fa",
                font=("Consolas", 9, "bold"),
            )

        active_x1 = record.X.matrix[0][0]
        active_x2 = record.X.matrix[0][1]

        for i in range(self.X.shape()[0]):
            px1 = self.X.matrix[i][0]
            px2 = self.X.matrix[i][1]
            t_val = self.T.matrix[i][self.output_index]
            u, v = self._world_to_pixel(px1, px2)

            if abs(px1 - active_x1) < 1e-6 and abs(px2 - active_x2) < 1e-6:
                self.canvas.create_oval(
                    u - 16,
                    v - 16,
                    u + 16,
                    v + 16,
                    outline="#cba6f7",
                    width=2,
                    dash=(3, 2),
                )

            if t_val > 0.0:
                self.canvas.create_oval(
                    u - 10, v - 10, u + 10, v + 10, fill="#f38ba8", outline="#11111b"
                )
                self.canvas.create_text(
                    u, v, text="+", fill="#11111b", font=("Consolas", 12, "bold")
                )
            else:
                self.canvas.create_oval(
                    u - 10, v - 10, u + 10, v + 10, fill="#a6e3a1", outline="#11111b"
                )
                self.canvas.create_text(
                    u, v, text="-", fill="#11111b", font=("Consolas", 12, "bold")
                )

        t_cur = record.target.matrix[0][self.output_index]
        yni_cur = record.net_input.matrix[0][self.output_index]
        y_cur = record.output.matrix[0][self.output_index]

        self.info_label.config(
            text=(
                f"Epoch {record.epoch} | Sample #{record.step_index + 1}: "
                f"x=[{active_x1:+.1f}, {active_x2:+.1f}], t={t_cur:+.1f}  ->  "
                f"y_NI={yni_cur:+.2f}, y={y_cur:+.1f} | "
                f"W=[{w1:+.2f}, {w2:+.2f}], b={b_val:+.2f}"
            )
        )

        if not any_line_drawn:
            eq_str = "Decision Boundary: Degenerate (W = [0, 0]) or outside viewport"
        elif theta > 1e-9:
            eq_str = (
                f"Separating Band: {b_val:+.2f} {w1:+.2f}*x1 {w2:+.2f}*x2 = ±{theta:.2f} "
                f"(Yellow: +θ, Orange: -θ)"
            )
        else:
            eq_str = f"Hyperplane Equation: {b_val:+.2f} {w1:+.2f}*x1 {w2:+.2f}*x2 = 0"

        self.eq_label.config(text=eq_str)
        total = len(self.snapshot_cb.records)
        self.step_counter_label.config(
            text=f"Step {self.current_idx + 1} of {total} (Use Left/Right Arrow Keys)"
        )

    def prev_step(self) -> None:
        if self.current_idx > 0:
            self.current_idx -= 1
            self._render_current_step()

    def next_step(self) -> None:
        if self.current_idx < len(self.snapshot_cb.records) - 1:
            self.current_idx += 1
            self._render_current_step()

    def first_step(self) -> None:
        self.current_idx = 0
        self._render_current_step()

    def last_step(self) -> None:
        self.current_idx = len(self.snapshot_cb.records) - 1
        self._render_current_step()

    def show(self) -> None:
        self.root.mainloop()