from __future__ import annotations

import json
import random
import tkinter as tk
from pathlib import Path

from ecit.layers.dense import DenseLayer
from ecit.math.matrix import Matrix

PRESETS: dict[str, list[str]] = {
    "X": [
        "#......#",
        ".#....#.",
        "..#..#..",
        "...##...",
        "...##...",
        "..#..#..",
        ".#....#.",
        "#......#",
    ],
    "O": [
        ".######.",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        ".######.",
    ],
}


class CharacterGridGUI:
    
    def __init__(
        self,
        layer: DenseLayer | None = None,
        rows: int = 5,
        cols: int = 5,
        dataset_path: str | Path | None = None,
        class_map: dict[int, str] | None = None,
        presets: dict[str, list[str]] | None = None,
        title: str = "ECIT - Interactive Character Grid",
    ) -> None:
        if layer is not None and rows * cols != layer.in_features:
            raise ValueError(
                f"Grid dimensions ({rows}x{cols}={rows * cols}) must match "
                f"layer.in_features ({layer.in_features})"
            )

        self.layer = layer
        self.rows = rows
        self.cols = cols
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None
        self.presets = presets if presets is not None else PRESETS
        self.title = title

        if class_map is not None:
            self.class_map = class_map
        elif layer is not None and layer.out_features == 1:
            self.class_map = {1: "X", -1: "O", 0: "Undecided (Dead-Zone)"}
        else:
            self.class_map = {0: "O (Unit Y1)", 1: "X (Unit Y2)"}

        self.grid_state: list[list[float]] = [
            [-1.0 for _ in range(cols)] for _ in range(rows)
        ]
        self.buttons: list[list[tk.Button]] = []

        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        self._build_ui()
        if "X" in self.presets:
            self.load_preset("X")
        else:
            self.evaluate_current_grid()

    def _build_ui(self) -> None:
        header = tk.Label(
            self.root,
            text=self.title,
            font=("Consolas", 12, "bold"),
            bg="#1e1e2e",
            fg="#cdd6f4",
            pady=10,
        )
        header.pack(fill=tk.X)

        grid_frame = tk.Frame(self.root, bg="#11111b", padx=10, pady=10)
        grid_frame.pack(padx=20, pady=6)

        for r in range(self.rows):
            btn_row: list[tk.Button] = []
            for c in range(self.cols):
                btn = tk.Button(
                    grid_frame,
                    text=".",
                    width=4,
                    height=2,
                    font=("Consolas", 14, "bold"),
                    relief=tk.RAISED,
                    bd=2,
                    command=lambda row=r, col=c: self.toggle_cell(row, col),
                )
                btn.grid(row=r, column=c, padx=3, pady=3)
                btn_row.append(btn)
            self.buttons.append(btn_row)

        ctrl_frame = tk.Frame(self.root, bg="#1e1e2e", pady=6)
        ctrl_frame.pack(fill=tk.X, padx=20)

        btn_cfg = {
            "font": ("Consolas", 10, "bold"),
            "bg": "#313244",
            "fg": "#cdd6f4",
            "activebackground": "#45475a",
            "activeforeground": "#ffffff",
            "relief": tk.FLAT,
            "padx": 10,
            "pady": 5,
        }

        for preset_name in self.presets:
            btn = tk.Button(
                ctrl_frame,
                text=f"Load '{preset_name}'",
                command=lambda name=preset_name: self.load_preset(name),
                **btn_cfg,
            )
            btn.pack(side=tk.LEFT, padx=4)

        noise_btn = tk.Button(
            ctrl_frame,
            text="Flip 1 Random Cell",
            command=self.inject_single_cell_noise,
            **btn_cfg,
        )
        noise_btn.pack(side=tk.LEFT, padx=4)

        clear_btn = tk.Button(
            ctrl_frame,
            text="Clear Grid",
            command=self.clear_grid,
            **btn_cfg,
        )
        clear_btn.pack(side=tk.RIGHT, padx=4)

        if self.dataset_path is not None:
            save_frame = tk.Frame(self.root, bg="#1e1e2e", pady=6)
            save_frame.pack(fill=tk.X, padx=20)

            save_x_btn = tk.Button(
                save_frame,
                text="Save as X (+1 / Class 1)",
                font=("Consolas", 10, "bold"),
                bg="#a6e3a1",
                fg="#11111b",
                activebackground="#94e2d5",
                relief=tk.FLAT,
                padx=12,
                pady=6,
                command=lambda: self.save_current_sample("X"),
            )
            save_x_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 6))

            save_o_btn = tk.Button(
                save_frame,
                text="Save as O (-1 / Class 0)",
                font=("Consolas", 10, "bold"),
                bg="#fab387",
                fg="#11111b",
                activebackground="#f9e2af",
                relief=tk.FLAT,
                padx=12,
                pady=6,
                command=lambda: self.save_current_sample("O"),
            )
            save_o_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(6, 4))

        status_frame = tk.Frame(
            self.root,
            bg="#181825",
            highlightthickness=1,
            highlightbackground="#45475a",
            padx=14,
            pady=10,
        )
        status_frame.pack(fill=tk.X, padx=20, pady=12)

        self.pred_label = tk.Label(
            status_frame,
            text="",
            font=("Consolas", 13, "bold"),
            bg="#181825",
            fg="#a6e3a1",
        )
        self.pred_label.pack(anchor=tk.W)

        self.telemetry_label = tk.Label(
            status_frame,
            text="",
            font=("Consolas", 10),
            bg="#181825",
            fg="#89b4fa",
            pady=4,
        )
        self.telemetry_label.pack(anchor=tk.W)

    def _sync_button_styles(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid_state[r][c]
                btn = self.buttons[r][c]
                if val > 0.0:
                    btn.config(
                        text="#",
                        bg="#89b4fa",
                        fg="#11111b",
                        activebackground="#b4befe",
                    )
                else:
                    btn.config(
                        text=".",
                        bg="#313244",
                        fg="#6c7086",
                        activebackground="#45475a",
                    )

    def _current_grid_as_strings(self) -> list[str]:
        return [
            "".join("#" if self.grid_state[r][c] > 0.0 else "." for c in range(self.cols))
            for r in range(self.rows)
        ]

    def toggle_cell(self, row: int, col: int) -> None:
        self.grid_state[row][col] *= -1.0
        self._sync_button_styles()
        self.evaluate_current_grid()

    def load_preset(self, name: str) -> None:
        template = self.presets[name]
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid_state[r][c] = 1.0 if template[r][c] == "#" else -1.0
        self._sync_button_styles()
        self.evaluate_current_grid()

    def clear_grid(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid_state[r][c] = -1.0
        self._sync_button_styles()
        self.evaluate_current_grid()

    def inject_single_cell_noise(self) -> None:
        r = random.randrange(self.rows)
        c = random.randrange(self.cols)
        self.toggle_cell(r, c)

    def save_current_sample(self, label: str) -> None:
        if self.dataset_path is None:
            return

        label_clean = label.upper()
        if label_clean == "X":
            bipolar_target = 1.0
            class_index = 1
        elif label_clean == "O":
            bipolar_target = -1.0
            class_index = 0
        else:
            raise ValueError(f"Unsupported label '{label}'. Expected 'X' or 'O'.")

        if self.dataset_path.exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("rows") != self.rows or data.get("cols") != self.cols:
                raise ValueError(
                    f"Existing dataset dimensions ({data.get('rows')}x{data.get('cols')}) "
                    f"do not match current GUI grid ({self.rows}x{self.cols})"
                )
        else:
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "rows": self.rows,
                "cols": self.cols,
                "classes": {
                    "O": {"bipolar_target": -1.0, "class_index": 0},
                    "X": {"bipolar_target": 1.0, "class_index": 1},
                },
                "samples": [],
            }

        samples: list[dict] = data["samples"]
        new_id = len(samples) + 1
        new_sample = {
            "id": new_id,
            "label": label_clean,
            "bipolar_target": bipolar_target,
            "class_index": class_index,
            "grid": self._current_grid_as_strings(),
        }
        samples.append(new_sample)

        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        x_count = sum(1 for s in samples if s["label"] == "X")
        o_count = sum(1 for s in samples if s["label"] == "O")

        self.pred_label.config(
            text=f"Saved Sample #{new_id} as '{label_clean}'!",
            fg="#a6e3a1" if label_clean == "X" else "#fab387",
        )
        self.telemetry_label.config(
            text=f"File: {self.dataset_path.name}  |  Total Samples: {len(samples)} (X: {x_count}, O: {o_count})"
        )

    def evaluate_current_grid(self) -> None:
        if self.layer is None:
            active_cells = sum(
                1
                for r in range(self.rows)
                for c in range(self.cols)
                if self.grid_state[r][c] > 0.0
            )
            self.pred_label.config(
                text="Dataset Builder Mode (Draw pattern & click Save)", fg="#cdd6f4"
            )
            file_info = (
                f"Target File: {self.dataset_path.name}"
                if self.dataset_path
                else "No file configured"
            )
            self.telemetry_label.config(
                text=f"Active '#' Cells: {active_cells}/{self.rows * self.cols}   |   {file_info}"
            )
            return

        flat_input = [
            self.grid_state[r][c] for r in range(self.rows) for c in range(self.cols)
        ]
        x_mat = Matrix.from_row(flat_input)
        y_mat = self.layer.forward(x_mat)
        yni_mat = self.layer.last_net_input
        assert yni_mat is not None

        if self.layer.out_features == 1:
            yni_val = yni_mat.item()
            y_val = y_mat.item()
            key = round(y_val)
            label = self.class_map.get(key, f"Class {key}")
            color = "#a6e3a1" if key != 0 else "#f9e2af"

            self.pred_label.config(text=f"Predicted Pattern: {label}", fg=color)
            self.telemetry_label.config(
                text=f"Net Input (y_NI) = {yni_val:+.2f}   |   Activated Output (y) = {y_val:+.1f}"
            )
        else:
            yni_list = yni_mat.matrix[0]
            y_list = y_mat.matrix[0]

            positive_indices = [j for j, val in enumerate(y_list) if val > 0.5]
            undecided_indices = [j for j, val in enumerate(y_list) if abs(val) <= 0.5]

            if len(positive_indices) == 1 and len(undecided_indices) == 0:
                winner = positive_indices[0]
                label = self.class_map.get(winner, f"Class {winner}")
                color = "#a6e3a1"
            else:
                label = "Ambiguous / Undecided"
                color = "#f38ba8"

            yni_str = "[" + ", ".join(f"{v:+.2f}" for v in yni_list) + "]"
            y_str = "[" + ", ".join(f"{v:+.1f}" for v in y_list) + "]"
            self.pred_label.config(text=f"Predicted Pattern: {label}", fg=color)
            self.telemetry_label.config(
                text=f"Net Input Vector y_NI = {yni_str}   |   Output Vector y = {y_str}"
            )

    def show(self) -> None:
        self.root.mainloop()