"""Floor plan and building section visualization from BuildingDesign."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

from car.models.design import BuildingDesign, WindowSize
from car.models.results import DesignIteration


class DesignPlotter:
    """Generates simplified floor plan sketches and building sections."""

    def plot_floor_plan(
        self,
        design: BuildingDesign,
        output_path: str | Path = "floor_plan.png",
        figsize: tuple[int, int] = (12, 10),
    ) -> None:
        """Render a 2D floor plan with building footprint and annotations."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        w = design.footprint_width_m
        d = design.footprint_depth_m

        # Building footprint
        footprint = Rectangle(
            (0, 0), w, d,
            linewidth=2, edgecolor="#333333",
            facecolor="#F5F5F0", zorder=2,
        )
        ax.add_patch(footprint)

        # Wall hatching based on wall type
        wall_color = {
            "load_bearing": "#8B7355",
            "curtain_wall": "#87CEEB",
            "partition": "#D2B48C",
        }
        color = wall_color.get(design.wall_type.value, "#8B7355")
        thickness_m = design.wall_thickness_mm / 1000.0

        # Draw walls (outer boundary with thickness)
        for side in ["bottom", "top", "left", "right"]:
            if side == "bottom":
                wall = Rectangle((0, 0), w, thickness_m, facecolor=color, zorder=3)
            elif side == "top":
                wall = Rectangle((0, d - thickness_m), w, thickness_m, facecolor=color, zorder=3)
            elif side == "left":
                wall = Rectangle((0, 0), thickness_m, d, facecolor=color, zorder=3)
            elif side == "right":
                wall = Rectangle((w - thickness_m, 0), thickness_m, d, facecolor=color, zorder=3)
            ax.add_patch(wall)

        # Windows (blue lines on walls based on window size)
        window_ratio = {
            "small": 0.15, "medium": 0.3, "large": 0.5, "full_glass": 0.75,
        }
        ratio = window_ratio.get(design.window_size.value, 0.3)

        # Draw windows on south-facing wall (bottom) and orientation wall
        window_width = w * ratio
        window_start = (w - window_width) / 2
        ax.plot(
            [window_start, window_start + window_width],
            [0, 0],
            color="#4169E1", linewidth=4, zorder=5,
        )
        ax.plot(
            [window_start, window_start + window_width],
            [d, d],
            color="#4169E1", linewidth=4, zorder=5,
        )

        # Compass rose
        compass_x, compass_y = w + 2, d / 2
        arrow_len = 1.5
        ax.annotate(
            "N", xy=(compass_x, compass_y + arrow_len),
            fontsize=12, fontweight="bold", ha="center",
        )
        ax.annotate(
            "", xy=(compass_x, compass_y + arrow_len),
            xytext=(compass_x, compass_y),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )

        # Dimension annotations
        ax.annotate(
            f"{w:.1f}m", xy=(w / 2, -1.5),
            fontsize=10, ha="center", color="#333333",
        )
        ax.annotate(
            f"{d:.1f}m", xy=(-1.5, d / 2),
            fontsize=10, ha="center", rotation=90, color="#333333",
        )

        # Info box
        info_text = (
            f"Structural: {design.structural_system.value.replace('_', ' ').title()}\n"
            f"Floors: {design.num_floors} ({design.building_height_m:.1f}m)\n"
            f"Wall: {design.wall_type.value.replace('_', ' ').title()} "
            f"({design.wall_thickness_mm:.0f}mm)\n"
            f"Windows: {design.window_size.value.title()}\n"
            f"Roof: {design.roof_type.value.replace('_', ' ').title()}\n"
            f"Floor Area: {design.floor_area_sqm:.0f} sqm"
        )
        ax.text(
            w + 1, -1, info_text,
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFDD", alpha=0.9),
            verticalalignment="top",
        )

        ax.set_xlim(-3, w + 8)
        ax.set_ylim(-3, d + 3)
        ax.set_aspect("equal")
        ax.set_title("Floor Plan (Ground Level)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Width (m)")
        ax.set_ylabel("Depth (m)")

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()

    def plot_building_section(
        self,
        design: BuildingDesign,
        output_path: str | Path = "building_section.png",
        figsize: tuple[int, int] = (10, 8),
    ) -> None:
        """Render a building cross-section showing floors and roof."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        w = design.footprint_width_m
        h = design.building_height_m
        n = design.num_floors
        floor_h = h / max(n, 1)

        # Building outline
        building = Rectangle(
            (0, 0), w, h,
            linewidth=2, edgecolor="#333333", facecolor="#F5F5F0",
        )
        ax.add_patch(building)

        # Floor slabs
        for i in range(n + 1):
            y = i * floor_h
            ax.plot([0, w], [y, y], color="#333333", linewidth=1.5)
            if i < n:
                ax.text(
                    w + 0.5, y + floor_h / 2,
                    f"Floor {i}",
                    fontsize=8, va="center",
                )

        # Roof profile
        roof_type = design.roof_type.value
        if roof_type == "pitched":
            roof_peak = h + 2.0
            ax.plot([0, w / 2, w], [h, roof_peak, h], color="#8B4513", linewidth=3)
            ax.fill([0, w / 2, w], [h, roof_peak, h], color="#CD853F", alpha=0.5)
        elif roof_type == "green_roof":
            green = Rectangle((0, h), w, 0.5, facecolor="#228B22", alpha=0.7)
            ax.add_patch(green)
            ax.text(w / 2, h + 0.25, "Green Roof", fontsize=8, ha="center", color="white")
        else:
            ax.plot([0, w], [h, h], color="#333333", linewidth=3)

        # Height annotation
        ax.annotate(
            "", xy=(-1, h), xytext=(-1, 0),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
        )
        ax.text(-2, h / 2, f"{h:.1f}m", fontsize=10, ha="center", rotation=90, color="red")

        # Wall thickness indication
        thick_m = design.wall_thickness_mm / 1000.0
        wall_l = Rectangle((0, 0), thick_m, h, facecolor="#8B7355", alpha=0.5)
        wall_r = Rectangle((w - thick_m, 0), thick_m, h, facecolor="#8B7355", alpha=0.5)
        ax.add_patch(wall_l)
        ax.add_patch(wall_r)

        ax.set_xlim(-4, w + 5)
        ax.set_ylim(-1, h + 4)
        ax.set_aspect("equal")
        ax.set_title("Building Cross-Section", fontsize=14, fontweight="bold")
        ax.set_xlabel("Width (m)")
        ax.set_ylabel("Height (m)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()

    def plot_design_comparison(
        self,
        designs: list[DesignIteration],
        output_path: str | Path = "design_comparison.png",
        max_designs: int = 6,
    ) -> None:
        """Render a grid comparing top N designs side-by-side."""
        n = min(len(designs), max_designs)
        if n == 0:
            return

        cols = min(n, 3)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

        if n == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx in range(n):
            row, col = divmod(idx, cols)
            ax = axes[row, col]
            d = designs[idx]
            bd = d.design

            # Simple box representation
            box = Rectangle(
                (0, 0), bd.footprint_width_m, bd.building_height_m,
                linewidth=2, edgecolor="#333333",
                facecolor="#D9B3FF" if d.compliance.is_compliant else "#FFB3B3",
                alpha=0.6,
            )
            ax.add_patch(box)

            ax.set_xlim(-2, bd.footprint_width_m + 2)
            ax.set_ylim(-2, bd.building_height_m + 4)
            ax.set_aspect("equal")

            title = (
                f"#{d.iteration_id} | Score: {d.overall_score:.2f}\n"
                f"{bd.structural_system.value.replace('_', ' ')}\n"
                f"{bd.num_floors}F | {bd.window_size.value} win | "
                f"{'PASS' if d.compliance.is_compliant else 'FAIL'}"
            )
            ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.2)

        # Hide unused axes
        for idx in range(n, rows * cols):
            row, col = divmod(idx, cols)
            axes[row, col].set_visible(False)

        fig.suptitle("Design Alternatives Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close()
