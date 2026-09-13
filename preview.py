"""Render what a model will look like before committing to a long print.

Draws the space-time structure directly: a dot per live cell, a line per rung.
Time is the vertical axis, so the picture is the printed object seen from the
side.

    python preview.py --pattern two_glider --frames 17 --out preview.png
    python preview.py --array my_pattern.txt --frames 22 --out mine.png
    python preview.py --seeds trees.npz --frames 22 --out trees.png
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import init_grids
from conway3d_stl import (life_run, plan_structure, load_grid_from_image,
                          load_building_blocks)

CELL_COLOR = "#8a5a34"
RUNG_COLOR = "#4a7f4a"


def part_extent(model_stl_dir="./model_stls/version3"):
    """Half-size of a cell and the depth of the base, to report true sizes."""
    b = load_building_blocks(model_stl_dir)
    cell = b["cell"].points.reshape(-1, 3)
    base_z = float(b["base"].points.reshape(-1, 3)[:, 2].min())
    return float(np.abs(cell).max()), base_z


def draw(ax, grid_record, unit=10, title="", base_radius=None, elev=14, azim=35,
         cell_half=3.575, base_z=-4.0):
    plan = plan_structure(grid_record)
    cells = np.array(plan["cells"], dtype=float)
    if len(cells) == 0:
        ax.set_title(f"{title} (empty)", fontsize=8)
        return None

    segs = [[(a[0] * unit, a[1] * unit, a[2] * unit),
             (b[0] * unit, b[1] * unit, b[2] * unit)]
            for a, b in plan["connection_lines"]]
    ax.add_collection3d(Line3DCollection(segs, colors=RUNG_COLOR,
                                         linewidths=0.6, alpha=0.85))
    xyz = cells * unit
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=6, c=CELL_COLOR,
               depthshade=False)

    # same centre and reach that make_circular_base uses
    lo2, hi2 = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
    cx, cy = (lo2 + hi2) / 2
    reach = np.linalg.norm(xyz[:, :2] - [cx, cy], axis=1).max()
    if base_radius == "auto":
        base_radius = reach + cell_half + 2.0
    if base_radius:
        ang = np.linspace(0, 2 * np.pi, 120)
        ax.plot(cx + base_radius * np.cos(ang), cy + base_radius * np.sin(ang),
                np.full_like(ang, -4.0), color="#999999", lw=1.0)
        reach = max(reach, base_radius)

    r = reach * 1.05 + unit
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(-4, max(xyz[:, 2].max(), 1) + unit)
    ax.set_box_aspect((1, 1, 1.6))
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=8)
    return {"cells": len(cells), "rungs": len(segs), "base_radius": base_radius,
            "height": float(xyz[:, 2].max() + cell_half - base_z),
            "width": float((reach + cell_half) * 2)}


def grid_from_args(args):
    if args.pattern:
        return np.asarray(getattr(init_grids, args.pattern))
    if args.array:
        return init_grids.load_pattern(args.array)
    return load_grid_from_image(args.image)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pattern")
    src.add_argument("--image")
    src.add_argument("--array", help="pattern file of on/off cells "
                                     "(text, .npy or .npz)")
    src.add_argument("--seeds", help=".npz of seed grids, one subplot each")
    p.add_argument("--frames", type=int, default=22)
    p.add_argument("--boundary", choices=["grow", "wall"], default="grow",
                   help="let the pattern spread past the edge of the grid, or "
                        "cut it off there")
    p.add_argument("--unit", type=float, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--base-radius", default=None,
                   help="plate radius in mm, or 'auto' to match --circular-base")
    p.add_argument("--elev", type=float, default=14)
    p.add_argument("--azim", type=float, default=35)
    args = p.parse_args()

    cell_half, base_z = part_extent()

    if args.seeds:
        d = np.load(args.seeds)
        keys = list(d.files)
        cols = min(args.cols, len(keys))
        rows = int(np.ceil(len(keys) / cols))
        fig = plt.figure(figsize=(3.1 * cols, 3.5 * rows))
        for i, k in enumerate(keys, 1):
            ax = fig.add_subplot(rows, cols, i, projection="3d")
            rec = life_run(np.asarray(d[k]), args.frames, boundary=args.boundary)
            info = draw(ax, rec, args.unit, title=k, elev=args.elev,
                        azim=args.azim, cell_half=cell_half, base_z=base_z,
                        base_radius=args.base_radius)
            if info:
                ax.set_title(f"{k}\n{info['cells']} cells, "
                             f"{info['height']:.0f}mm tall, "
                             f"{info['base_radius'] * 2:.0f}mm base"
                             if info.get("base_radius") else
                             f"{k}\n{info['cells']} cells, "
                             f"{info['height']:.0f}mm tall, "
                             f"{info['width']:.0f}mm wide", fontsize=7)
    else:
        rec = life_run(grid_from_args(args), args.frames, boundary=args.boundary)
        fig = plt.figure(figsize=(6, 7))
        ax = fig.add_subplot(projection="3d")
        info = draw(ax, rec, args.unit,
                    title=args.pattern or args.array or args.image,
                    base_radius=args.base_radius, elev=args.elev, azim=args.azim,
                    cell_half=cell_half, base_z=base_z)
        if info:
            print(f"{info['cells']} cells, {info['rungs']} rungs, "
                  f"{info['height']:.0f}mm tall, {info['width']:.0f}mm wide")

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
