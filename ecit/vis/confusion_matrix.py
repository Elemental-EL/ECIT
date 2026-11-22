import math
import tkinter as tk


def sum_confusion_matrices(matrices: list[list[list[int]]]) -> list[list[int]]:
    if not matrices:
        raise ValueError("Cannot sum an empty list of confusion matrices")
    k = len(matrices[0])
    total = [[0 for _ in range(k)] for _ in range(k)]
    for cm in matrices:
        if len(cm) != k or any(len(row) != k for row in cm):
            raise ValueError("All confusion matrices must be square with identical dimensions")
        for r in range(k):
            for c in range(k):
                total[r][c] += cm[r][c]
    return total


class ConfusionMatrixPlotter:
    def __init__(
        self,
        matrices: list[list[int]] | dict[str, list[list[int]]],
        class_names: list[str] | None = None,
        title: str = "ECIT - Confusion Matrix",
        cell_width: int = 92,
        cell_height: int = 64,
    ) -> None:
        if isinstance(matrices, list):
            self.panels: dict[str, list[list[int]]] = {"Confusion Matrix": matrices}
        elif isinstance(matrices, dict) and len(matrices) > 0:
            self.panels = matrices
        else:
            raise ValueError("matrices must be a 2D list or a non-empty dict of {title: 2D list}")

        first_cm = next(iter(self.panels.values()))
        self.num_classes = len(first_cm)
        if self.num_classes < 2:
            raise ValueError("Confusion matrix must have at least 2 classes")

        for name, cm in self.panels.items():
            if len(cm) != self.num_classes or any(len(row) != self.num_classes for row in cm):
                raise ValueError(
                    f"Matrix '{name}' must be square of shape ({self.num_classes}, {self.num_classes})"
                )

        if class_names is None:
            self.class_names = [str(i + 1) for i in range(self.num_classes)]
        else:
            if len(class_names) != self.num_classes:
                raise ValueError(
                    f"len(class_names)={len(class_names)} must match matrix size {self.num_classes}"
                )
            self.class_names = list(class_names)

        self.title = title
        self.cell_w = cell_width
        self.cell_h = cell_height

        self.pad_left = 105
        self.pad_top = 44
        self.pad_right = 24
        self.pad_bottom = 64

        self.panel_w = self.pad_left + (self.num_classes + 1) * self.cell_w + self.pad_right
        self.panel_h = self.pad_top + (self.num_classes + 1) * self.cell_h + self.pad_bottom

        self.grid_cols = 1 if len(self.panels) == 1 else 2
        self.grid_rows = math.ceil(len(self.panels) / self.grid_cols)

        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        self._build_ui()

    def _build_ui(self) -> None:
        header = tk.Label(
            self.root,
            text=self.title,
            font=("Consolas", 13, "bold"),
            bg="#1e1e2e",
            fg="#cdd6f4",
            pady=10,
        )
        header.pack(fill=tk.X)

        container = tk.Frame(self.root, bg="#1e1e2e", padx=12, pady=4)
        container.pack()

        for idx, (panel_title, cm) in enumerate(self.panels.items()):
            r = idx // self.grid_cols
            c = idx % self.grid_cols
            canvas = tk.Canvas(
                container,
                width=self.panel_w,
                height=self.panel_h,
                bg="#181825",
                highlightthickness=1,
                highlightbackground="#45475a",
            )
            canvas.grid(row=r, column=c, padx=8, pady=8)
            self._draw_panel(canvas, panel_title, cm)

        legend = tk.Label(
            self.root,
            text="Green: Correct / Precision / Recall   |   Red: Misclassified / Error   |   Blue: Overall Accuracy",
            font=("Consolas", 9),
            bg="#1e1e2e",
            fg="#a6adc8",
            pady=8,
        )
        legend.pack(fill=tk.X)

    def _draw_panel(self, canvas: tk.Canvas, panel_title: str, cm: list[list[int]]) -> None:
        k = self.num_classes
        x0 = self.pad_left
        y0 = self.pad_top
        grid_w = (k + 1) * self.cell_w
        grid_h = (k + 1) * self.cell_h

        total = sum(sum(row) for row in cm)
        row_sums = [sum(cm[r]) for r in range(k)]
        col_sums = [sum(cm[r][c] for r in range(k)) for c in range(k)]
        diag_sum = sum(cm[i][i] for i in range(k))

        canvas.create_text(
            x0 + grid_w / 2,
            22,
            text=panel_title,
            fill="#cdd6f4",
            font=("Consolas", 11, "bold"),
        )

        for r in range(k):
            for c in range(k):
                cx1 = x0 + c * self.cell_w
                cy1 = y0 + r * self.cell_h
                cx2 = cx1 + self.cell_w
                cy2 = cy1 + self.cell_h

                count = cm[r][c]
                pct = (100.0 * count / total) if total > 0 else 0.0

                if r == c:
                    fill_bg = "#2d5a3c"
                    pct_color = "#a6e3a1"
                elif count > 0:
                    fill_bg = "#632b38"
                    pct_color = "#f38ba8"
                else:
                    fill_bg = "#35222b"
                    pct_color = "#a6adc8"

                canvas.create_rectangle(cx1, cy1, cx2, cy2, fill=fill_bg, outline="#11111b", width=2)
                canvas.create_text(
                    (cx1 + cx2) / 2,
                    cy1 + self.cell_h * 0.36,
                    text=str(count),
                    fill="#ffffff",
                    font=("Consolas", 11, "bold"),
                )
                canvas.create_text(
                    (cx1 + cx2) / 2,
                    cy1 + self.cell_h * 0.70,
                    text=f"{pct:.1f}%",
                    fill=pct_color,
                    font=("Consolas", 9),
                )

        for r in range(k):
            cx1 = x0 + k * self.cell_w
            cy1 = y0 + r * self.cell_h
            cx2 = cx1 + self.cell_w
            cy2 = cy1 + self.cell_h

            canvas.create_rectangle(cx1, cy1, cx2, cy2, fill="#313244", outline="#11111b", width=2)
            if row_sums[r] > 0:
                prec = 100.0 * cm[r][r] / row_sums[r]
                fdr = 100.0 - prec
                top_txt, bot_txt = f"{prec:.1f}%", f"{fdr:.1f}%"
            else:
                top_txt, bot_txt = "N/A", "N/A"

            canvas.create_text(
                (cx1 + cx2) / 2,
                cy1 + self.cell_h * 0.34,
                text=top_txt,
                fill="#a6e3a1",
                font=("Consolas", 9, "bold"),
            )
            canvas.create_text(
                (cx1 + cx2) / 2,
                cy1 + self.cell_h * 0.68,
                text=bot_txt,
                fill="#f38ba8",
                font=("Consolas", 9),
            )

        for c in range(k):
            cx1 = x0 + c * self.cell_w
            cy1 = y0 + k * self.cell_h
            cx2 = cx1 + self.cell_w
            cy2 = cy1 + self.cell_h

            canvas.create_rectangle(cx1, cy1, cx2, cy2, fill="#313244", outline="#11111b", width=2)
            if col_sums[c] > 0:
                rec = 100.0 * cm[c][c] / col_sums[c]
                fnr = 100.0 - rec
                top_txt, bot_txt = f"{rec:.1f}%", f"{fnr:.1f}%"
            else:
                top_txt, bot_txt = "N/A", "N/A"

            canvas.create_text(
                (cx1 + cx2) / 2,
                cy1 + self.cell_h * 0.34,
                text=top_txt,
                fill="#a6e3a1",
                font=("Consolas", 9, "bold"),
            )
            canvas.create_text(
                (cx1 + cx2) / 2,
                cy1 + self.cell_h * 0.68,
                text=bot_txt,
                fill="#f38ba8",
                font=("Consolas", 9),
            )

        cx1 = x0 + k * self.cell_w
        cy1 = y0 + k * self.cell_h
        cx2 = cx1 + self.cell_w
        cy2 = cy1 + self.cell_h
        acc = (100.0 * diag_sum / total) if total > 0 else 0.0
        err = 100.0 - acc if total > 0 else 0.0

        canvas.create_rectangle(cx1, cy1, cx2, cy2, fill="#2a446e", outline="#11111b", width=2)
        canvas.create_text(
            (cx1 + cx2) / 2,
            cy1 + self.cell_h * 0.34,
            text=f"{acc:.1f}%",
            fill="#a6e3a1",
            font=("Consolas", 10, "bold"),
        )
        canvas.create_text(
            (cx1 + cx2) / 2,
            cy1 + self.cell_h * 0.68,
            text=f"{err:.1f}%",
            fill="#f38ba8",
            font=("Consolas", 9, "bold"),
        )

        for r in range(k):
            canvas.create_text(
                x0 - 10,
                y0 + (r + 0.5) * self.cell_h,
                text=self.class_names[r],
                anchor=tk.E,
                fill="#cdd6f4",
                font=("Consolas", 9, "bold"),
            )

        for c in range(k):
            canvas.create_text(
                x0 + (c + 0.5) * self.cell_w,
                y0 + grid_h + 14,
                text=self.class_names[c],
                anchor=tk.N,
                fill="#cdd6f4",
                font=("Consolas", 9, "bold"),
            )

        canvas.create_text(
            20,
            y0 + (k * self.cell_h) / 2,
            text="Output Class",
            angle=90,
            fill="#89b4fa",
            font=("Consolas", 10, "bold"),
        )
        canvas.create_text(
            x0 + (k * self.cell_w) / 2,
            y0 + grid_h + 44,
            text="Target Class",
            fill="#89b4fa",
            font=("Consolas", 10, "bold"),
        )

    def show(self) -> None:
        self.root.mainloop()