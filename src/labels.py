"""A small label placer, because matplotlib will happily stack ten names on top
of each other and call it a scatterplot.

Labels start on the side of the point facing away from the middle of the plot,
then get pushed apart vertically until nothing overlaps. A leader line is drawn
whenever a label ends up more than a nudge from its point.
"""

from __future__ import annotations

import numpy as np


def place(ax, xs, ys, texts, colour="#0b0b0b", fontsize=8.6, weight="semibold",
          pad_x=0.014, line_colour="#8a8880", passes=260):
    """Annotate points with names, resolving vertical collisions."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    fx = (xs - x0) / (x1 - x0)
    fy = (ys - y0) / (y1 - y0)
    right = fx <= 0.62      # name goes to the right unless the point is already there
    # Work in axis fractions, so the box size has to come back from pixels to points.
    dpi = ax.figure.dpi
    ax_w_pt = ax.get_window_extent().width / dpi * 72
    ax_h_pt = ax.get_window_extent().height / dpi * 72
    box_h = fontsize * 1.5 / ax_h_pt
    widths = np.array([len(t) for t in texts]) * fontsize * 0.56 / ax_w_pt

    ty = fy.copy()
    tx = np.where(right, fx + pad_x, fx - pad_x)

    for _ in range(passes):
        moved = False
        order = np.argsort(ty)
        for a_i in range(len(order)):
            for b_i in range(a_i + 1, len(order)):
                a, b = order[a_i], order[b_i]
                if right[a] != right[b]:
                    continue
                if abs(ty[a] - ty[b]) >= box_h:
                    break
                overlap_x = abs(tx[a] - tx[b]) < max(widths[a], widths[b]) * 0.9
                if not overlap_x:
                    continue
                shift = (box_h - abs(ty[a] - ty[b])) / 2 + 1e-4
                lo, hi = (a, b) if ty[a] <= ty[b] else (b, a)
                ty[lo] -= shift
                ty[hi] += shift
                moved = True
        ty = np.clip(ty, box_h / 2, 1 - box_h / 2)
        if not moved:
            break

    for i, text in enumerate(texts):
        px, py = tx[i], ty[i]
        if abs(py - fy[i]) > box_h * 0.55:
            ax.plot([fx[i] + (0.004 if right[i] else -0.004), px],
                    [fy[i], py], transform=ax.transAxes,
                    color=line_colour, lw=0.7, zorder=3)
        ax.text(px, py, text, transform=ax.transAxes,
                ha="left" if right[i] else "right", va="center",
                fontsize=fontsize, color=colour, fontweight=weight, zorder=6)
