from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Sequence


def _polyline(values: Sequence[float], x: float, y: float, width: float, height: float) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    spread = high - low or 1.0
    points = []
    for index, value in enumerate(values):
        px = x + index * width / max(1, len(values) - 1)
        py = y + height - (value - low) * height / spread
        points.append(f"{px:.1f},{py:.1f}")
    return " ".join(points)


def write_history_svg(history_csv: str | Path, destination: str | Path) -> Path:
    with Path(history_csv).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Training history is empty: {history_csv}")
    epochs = [int(row["epoch"]) for row in rows]
    series = {
        key: [float(row[key]) for row in rows]
        for key in ("train_loss", "validation_loss", "train_accuracy", "validation_accuracy")
    }
    width, height = 900, 520
    x, y, plot_width, plot_height = 80, 65, 760, 360
    colors = {
        "train_loss": "#2563eb",
        "validation_loss": "#dc2626",
        "train_accuracy": "#059669",
        "validation_accuracy": "#d97706",
    }
    lines = []
    for key, values in series.items():
        group_y = y if "loss" in key else y + 195
        group_h = 150
        poly = _polyline(values, x, group_y, plot_width, group_h)
        lines.append(
            f'<polyline fill="none" stroke="{colors[key]}" stroke-width="2.5" points="{poly}"/>'
        )
    legend = []
    for index, (key, color) in enumerate(colors.items()):
        legend.append(
            f'<text x="{80 + (index % 2) * 260}" y="{465 + (index // 2) * 20}" fill="{color}">'
            f'{html.escape(key.replace("_", " "))}</text>'
        )
    labels = "".join(
        f'<text x="{x + i * plot_width / max(1, len(epochs) - 1):.1f}" y="445" '
        f'text-anchor="middle" font-size="12">{epoch}</text>'
        for i, epoch in enumerate(epochs)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        f'<text x="80" y="32" font-size="20">Training and validation history</text>'
        f'<rect x="{x}" y="{y}" width="{plot_width}" height="150" fill="none" stroke="#aaa"/>'
        f'<rect x="{x}" y="{y + 195}" width="{plot_width}" height="150" fill="none" stroke="#aaa"/>'
        f'{"".join(lines)}{labels}{"".join(legend)}</svg>'
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path


def write_confusion_svg(
    labels: Sequence[str],
    counts: Sequence[Sequence[int]],
    destination: str | Path,
) -> Path:
    if len(labels) != len(counts) or any(len(row) != len(labels) for row in counts):
        raise ValueError("Confusion matrix dimensions must match the label count")
    cell = 54
    margin_left, margin_top = 150, 100
    size = margin_left + cell * len(labels) + 35
    maximum = max((value for row in counts for value in row), default=0) or 1
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-size="19">Confusion matrix (actual rows, predicted columns)</text>',
    ]
    for row_index, label in enumerate(labels):
        y = margin_top + row_index * cell
        content.append(
            f'<text x="{margin_left - 8}" y="{y + 33}" text-anchor="end" font-size="11">'
            f'{html.escape(label)}</text>'
        )
        for col_index, value in enumerate(counts[row_index]):
            x = margin_left + col_index * cell
            shade = round(255 - 175 * value / maximum)
            content.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="rgb({shade},{shade},{255})" stroke="white"/>'
            )
            content.append(
                f'<text x="{x + cell / 2}" y="{y + 32}" text-anchor="middle" font-size="12">'
                f'{value}</text>'
            )
    for col_index, label in enumerate(labels):
        x = margin_left + col_index * cell + cell / 2
        content.append(
            f'<text x="{x:.1f}" y="{margin_top - 8}" text-anchor="end" font-size="11" '
            f'transform="rotate(-45 {x:.1f} {margin_top - 8})">{html.escape(label)}</text>'
        )
    content.append("</svg>")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(content), encoding="utf-8")
    return path
