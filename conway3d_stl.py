"""Conway's Game of Life (2D + time) -> print-in-place 3D model.

The game is run forward from an initial grid, and the resulting space-time
history is turned into a single STL:

  * every live cell at (x, y, t) becomes a "cell" solid, placed on a regular
    lattice with spacing ``unit`` in x, y and t (t becomes the print's z axis);
  * wherever a live cell at t has a live neighbour at t+1, a "rung" solid is
    placed to link the two, so successive generations hold each other up;
  * cells alive at t=0 also get a flat "base" so the model has a footprint to
    print on.

Because the rungs are the only thing joining one generation to the next, the
model only prints as one piece if the space-time pattern stays connected --
see check_connectivity().

Usage:
    python conway3d_stl.py --pattern two_glider --frames 17 \
        --out output_stls/two_glider_pond.stl

    python conway3d_stl.py --image initial_sates/space_invader_3_cute.png \
        --frames 15 --out output_stls/space_invader.stl --plot
"""

import argparse
import os

import numpy as np
from numpy import arctan2
from scipy.signal import convolve2d as conv2d

import stl
from stl import mesh

import init_grids


# ---------------------------------------------------------------- game of life

conwayog_rule_mask = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]])


def conway_update(state, neighbors):
    if neighbors < 2 or neighbors > 3:
        state = 0
    elif neighbors == 3:
        state = 1
    return state


conway_updatev = np.vectorize(conway_update)


def life_run(grid_init, num_frames, rule_mask=conwayog_rule_mask):
    """Run the game of life for num_frames generations (frame 0 is the input)."""
    grid_history = np.zeros((num_frames, *grid_init.shape))
    grid_history[0, :, :] = grid_init

    for f in range(1, num_frames):
        grid_history[f, :, :] = conway_updatev(
            grid_history[f - 1, :, :],
            conv2d(grid_history[f - 1, :, :], rule_mask,
                   mode="same", boundary="fill"))

    return grid_history


def load_grid_from_image(path):
    """Load a 1-bit bitmap as a grid; black pixels are live cells."""
    from PIL import Image
    imagebmp = Image.open(path)
    return np.array(~np.array(imagebmp)).astype(int)


# ------------------------------------------------------------- building blocks

# Offsets of the eight neighbours a cell can be linked to in the next frame.
REGION_CHECK = [[1, 0], [1, 1], [0, 1], [-1, 1],
                [-1, 0], [-1, -1], [0, -1], [1, -1]]


def load_building_blocks(model_stl_dir="./model_stls/version3"):
    """Load the four part STLs and orient them for placement."""
    rungstl_adj = mesh.Mesh.from_file(f"{model_stl_dir}/rung_adj.stl")
    rungstl_crn = mesh.Mesh.from_file(f"{model_stl_dir}/rung_diag.stl")
    cellbasestl = mesh.Mesh.from_file(f"{model_stl_dir}/base.stl")
    cellstl = mesh.Mesh.from_file(f"{model_stl_dir}/cell.stl")

    # rotation to have each mesh oriented in positive x-axis
    rungstl_adj.rotate([0, 0, 1], np.pi / 2)
    rungstl_crn.rotate([0, 0, 1], 2 * np.pi / 4)
    cellbasestl.z -= 4  # the base cell center is at positive 4mm

    return {"rung_adj": rungstl_adj, "rung_crn": rungstl_crn,
            "base": cellbasestl, "cell": cellstl}


def plan_structure(grid_record, verbose=False):
    """Work out where every cell and rung goes, without building any geometry.

    Returns the (x, y, t) of every live cell and, for each link, the two cells
    it joins plus which rung part reaches between them.
    """
    cells = []
    connection_lines = []
    rungs = []

    grid_rp = np.pad(grid_record, ((0, 0), (1, 1), (1, 1)))  # grid record with padding
    grid_rp = grid_rp[:, ::-1, :]  # flipped, so when printed, the original pattern is seen from below

    im = 1
    for t in range(grid_rp.shape[0]):
        for x in range(1, grid_rp.shape[1]):  # for each valid grid point before padding
            for y in range(1, grid_rp.shape[2]):
                if grid_rp[t, x, y] != 1:
                    continue

                cells.append((x, y, t))

                if t < grid_rp.shape[0] - 1:
                    for [dx, dy] in REGION_CHECK:
                        if grid_rp[t + 1, x + dx, y + dy] != 1:
                            continue
                        # an orthogonal step needs the short rung, a diagonal
                        # step the long one
                        kind = "rung_adj" if (dx + dy) % 2 == 1 else "rung_crn"
                        if verbose:
                            print(f"{im} (t, x, y) : {(t, x, y)} to "
                                  f"{(t + 1, x + dx, y + dy)}, "
                                  f"{'adj' if kind == 'rung_adj' else 'crn'}")
                        rungs.append((x, y, t, dx, dy, kind))
                        connection_lines.append([[x, y, t], [x + dx, y + dy, t + 1]])
                        im += 1

    return {"cells": cells, "rungs": rungs, "connection_lines": connection_lines}


def build_model(grid_record, blocks, unit=10, verbose=False):
    """Place a part mesh for every cell and every cell-to-cell link."""
    plan = plan_structure(grid_record, verbose=verbose)
    meshes = []

    for (x, y, t) in plan["cells"]:
        m = mesh.Mesh(np.copy(blocks["cell"].data))  # mesh for each live cell
        m.x += x * unit
        m.y += y * unit
        m.z += t * unit
        meshes.append(mesh.Mesh(np.copy(m.data)))

        if t == 0:  # adding the base for printing
            m = mesh.Mesh(np.copy(blocks["base"].data))
            m.x += x * unit
            m.y += y * unit
            meshes.append(mesh.Mesh(np.copy(m.data)))

    for (x, y, t, dx, dy, kind) in plan["rungs"]:
        m = mesh.Mesh(np.copy(blocks[kind].data))
        m.rotate([0, 0, 1], -arctan2(dy, dx))  # rotation around z axis
        m.x += x * unit  # translation of x and y coordinates
        m.y += y * unit
        m.z += t * unit
        meshes.append(mesh.Mesh(np.copy(m.data)))

    return {"meshes": meshes, "cells": plan["cells"],
            "connection_lines": plan["connection_lines"], "unit": unit}


def check_connectivity(model):
    """Group cells into pieces joined by rungs.

    A print-in-place model needs every piece to reach the base, so this returns
    the list of connected components and which of them touch t=0.
    """
    parent = {c: c for c in model["cells"]}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for [x0, y0, t0], [x1, y1, t1] in model["connection_lines"]:
        union((x0, y0, t0), (x1, y1, t1))

    groups = {}
    for c in model["cells"]:
        groups.setdefault(find(c), []).append(c)

    components = sorted(groups.values(), key=len, reverse=True)
    grounded = [g for g in components if any(t == 0 for _, _, t in g)]
    return {"components": components, "grounded": grounded,
            "floating": [g for g in components if g not in grounded]}


def save_stl(meshes, path, ascii_mode=True):
    """Concatenate the placed meshes and write one STL."""
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    combined = mesh.Mesh(np.concatenate([m.data for m in meshes]))
    combined.save(path, mode=stl.Mode.ASCII if ascii_mode else stl.Mode.BINARY)
    return combined


# ------------------------------------------------------------- circular base

def cylinder_mesh(cx, cy, z0, z1, radius, segments=180):
    """A closed cylinder, used as the stabilising base plate."""
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    x = cx + radius * np.cos(ang)
    y = cy + radius * np.sin(ang)

    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        p0, p1 = [x[i], y[i], z0], [x[j], y[j], z0]
        q0, q1 = [x[i], y[i], z1], [x[j], y[j], z1]
        tris.append([[cx, cy, z0], p1, p0])          # bottom fan, faces -z
        tris.append([[cx, cy, z1], q0, q1])          # top fan, faces +z
        tris.append([p0, p1, q1])                    # side
        tris.append([p0, q1, q0])

    data = np.zeros(len(tris), dtype=mesh.Mesh.dtype)
    data["vectors"] = np.array(tris)
    m = mesh.Mesh(data)
    m.update_normals()
    return m


def base_footprint(model, blocks):
    """Centre and radius of the smallest disc covering the whole model in xy.

    A tree-shaped model is top heavy, so the plate is sized to the canopy
    rather than to the few cells that touch the plate.
    """
    unit = model["unit"]
    pts = np.array([[x * unit, y * unit] for x, y, _ in model["cells"]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    cell_r = float(np.abs(blocks["cell"].points.reshape(-1, 3)[:, :2]).max())
    radius = float(np.linalg.norm(pts - centre, axis=1).max()) + cell_r
    return centre[0], centre[1], radius


def make_circular_base(model, blocks, radius=None, thickness=3.0, segments=180,
                       margin=2.0):
    """Build the plate and return (mesh, centre_x, centre_y, radius).

    It spans from the underside of the per-cell bases upwards, so it fuses
    with every cell that touches the plate.
    """
    cx, cy, auto_r = base_footprint(model, blocks)
    if radius is None:
        radius = auto_r + margin
    z0 = float(blocks["base"].points.reshape(-1, 3)[:, 2].min())  # -4mm, the plate
    return cylinder_mesh(cx, cy, z0, z0 + thickness, radius, segments), cx, cy, radius


# ----------------------------------------------------------------- diagnostics

def plot_all(grid_record, model, blocks):
    """The original exploratory plots: final frame, voxels, links, meshes."""
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    plt.pcolormesh(grid_record[-1])
    plt.grid()
    plt.show()

    # Plot the game of life output as voxels, frame 0 in red and the rest green
    ax = plt.figure().add_subplot(projection='3d')
    grid_temp = np.zeros(grid_record.shape)
    grid_temp[[0], :, :] = grid_record[[0], :, :]
    ax.voxels(np.moveaxis(grid_temp, (0, 1, 2), (0, 2, 1))[:, ::-1, ::-1],
              facecolors="red", edgecolor='k')
    grid_temp = np.zeros(grid_record.shape)
    grid_temp[1:, :, :] = grid_record[1:, :, :]
    ax.voxels(np.moveaxis(grid_temp, (0, 1, 2), (0, 2, 1))[:, ::-1, ::-1],
              facecolors="green", edgecolor='k')
    ax.set_xlim(0, grid_record.shape[0])
    ax.set_ylim(0, grid_record.shape[1])
    ax.set_zlim(0, grid_record.shape[2])
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    plt.show()

    # plotting connections
    ax = plt.figure().add_subplot(projection='3d')
    for cl in model["connection_lines"]:
        [xs, ys, zs] = list(zip(*cl))
        ax.plot(xs, ys, zs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    plt.show()

    # plotting combined meshes.  Axes3D(figure) no longer attaches itself to the
    # figure (matplotlib >= 3.4), so build the 3d axes through the figure.
    figure = plt.figure()
    axes = figure.add_subplot(projection='3d')
    scale = blocks["rung_adj"].points.flatten()
    for m in model["meshes"]:
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
    axes.auto_scale_xyz(scale * 6, scale * 6, scale * 6)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("t")
    plt.show()


# ------------------------------------------------------------------------ main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pattern", help="name of a grid defined in init_grids.py")
    src.add_argument("--image", help="1-bit bitmap to use as the initial grid")
    p.add_argument("--frames", type=int, default=15,
                   help="number of generations, including the initial one")
    p.add_argument("--out", required=True, help="output STL path")
    p.add_argument("--model-stls", default="./model_stls/version3",
                   help="directory holding base/cell/rung_adj/rung_diag STLs")
    p.add_argument("--unit", type=float, default=10,
                   help="lattice spacing in mm, must match the part STLs")
    p.add_argument("--binary", action="store_true",
                   help="write a binary STL instead of ASCII")
    p.add_argument("--center", action="store_true",
                   help="move the model over the origin, ready to slice")
    p.add_argument("--circular-base", action="store_true",
                   help="add a round plate under the model so it cannot tip")
    p.add_argument("--base-radius", type=float, default=None,
                   help="plate radius in mm (default: cover the whole model)")
    p.add_argument("--base-thickness", type=float, default=3.0,
                   help="plate thickness in mm")
    p.add_argument("--base-segments", type=int, default=180,
                   help="facets around the plate")
    p.add_argument("--pad", type=int, default=0,
                   help="pad the initial grid with this many dead cells")
    p.add_argument("--plot", action="store_true", help="show the diagnostic plots")
    p.add_argument("--verbose", action="store_true", help="print every rung placed")
    args = p.parse_args(argv)

    if args.pattern:
        grid = getattr(init_grids, args.pattern, None)
        if grid is None:
            p.error(f"init_grids.py has no pattern named {args.pattern!r}")
        grid = np.asarray(grid)
    else:
        grid = load_grid_from_image(args.image)

    if args.pad:
        grid = np.pad(grid, ((args.pad, args.pad), (args.pad, args.pad)))

    grid_record = life_run(grid, args.frames)
    blocks = load_building_blocks(args.model_stls)
    model = build_model(grid_record, blocks, unit=args.unit, verbose=args.verbose)

    conn = check_connectivity(model)

    out_meshes = list(model["meshes"])
    plate = None
    if args.circular_base:
        plate, bx, by, br = make_circular_base(
            model, blocks, radius=args.base_radius,
            thickness=args.base_thickness, segments=args.base_segments)
        out_meshes.append(plate)

    if args.center:
        pts = np.concatenate([m.points.reshape(-1, 3) for m in out_meshes])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        shift = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])
        for m in out_meshes:
            m.x -= shift[0]
            m.y -= shift[1]

    combined = save_stl(out_meshes, args.out, ascii_mode=not args.binary)

    print(f"grid {grid.shape[0]}x{grid.shape[1]}, {args.frames} frames")
    print(f"live cells: {len(model['cells'])}, rungs: {len(model['connection_lines'])}")
    print(f"parts: {len(model['meshes'])}, triangles: {len(combined.data)}")
    print(f"connected pieces: {len(conn['components'])}"
          f" ({len(conn['floating'])} not reaching the base)")
    if plate is not None:
        print(f"circular base: radius {br:.1f}mm, {args.base_thickness}mm thick, "
              f"centred at ({bx:.1f}, {by:.1f})")
    print(f"wrote {args.out}")

    if args.plot:
        plot_all(grid_record, model, blocks)

    return model


if __name__ == "__main__":
    main()
