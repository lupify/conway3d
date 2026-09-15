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

    python conway3d_stl.py --image initial_states/space_invader_3_cute.png \
        --frames 15 --out output_stls/space_invader.stl --plot

    python conway3d_stl.py --array my_pattern.txt --frames 22 \
        --base plate --center --binary --out output_stls/mine.stl

    python conway3d_stl.py --pattern gosper_glider_gun --frames 40 \
        --base both --center --binary --out output_stls/gun.stl
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


def room_to_grow(grid, num_frames):
    """Enlarge a grid so num_frames generations cannot reach its edge.

    A pattern spreads by at most one cell per generation, so after num_frames
    generations nothing is further than num_frames - 1 from where it started.
    Leaving num_frames dead cells beyond the live cells therefore keeps the
    outermost ring dead for the whole run, which makes the neighbour counts
    exact and the result identical to Life on an unbounded plane.

    A grid that is already roomy enough is returned untouched.
    """
    pad = grow_pad(grid, num_frames)
    if not any(p for side in pad for p in side):
        return np.asarray(grid)
    return np.pad(np.asarray(grid), pad)


def grow_pad(grid, num_frames):
    """How much room_to_grow would add on each side, as ((top, bottom), (left, right))."""
    grid = np.asarray(grid)
    ys, xs = np.nonzero(grid)
    if len(ys) == 0:
        return ((0, 0), (0, 0))
    return ((max(0, num_frames - int(ys.min())),
             max(0, num_frames - int(grid.shape[0] - 1 - ys.max()))),
            (max(0, num_frames - int(xs.min())),
             max(0, num_frames - int(grid.shape[1] - 1 - xs.max()))))


def life_run(grid_init, num_frames, rule_mask=conwayog_rule_mask,
             boundary="grow"):
    """Run the game of life for num_frames generations (frame 0 is the input).

    boundary "wall" keeps the grid exactly as given and treats everything
    outside it as permanently dead, so a pattern reaching the edge is cut off.
    boundary "grow", the default, enlarges the grid as far as the run could
    possibly need, so the pattern is never cut off and the result is true
    unbounded Life.  Note that it therefore returns frames larger than the grid
    passed in, whenever the pattern starts near an edge.
    """
    if boundary == "grow":
        grid_init = room_to_grow(grid_init, num_frames)
    elif boundary != "wall":
        raise ValueError(f"boundary must be 'wall' or 'grow', not {boundary!r}")

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


def load_building_blocks(model_stl_dir="./model_stls/version3", unit=10):
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
            "base": cellbasestl, "cell": cellstl,
            "start": make_start_cell(cellstl, foot_radius=unit / 2)}


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


def build_model(grid_record, blocks, unit=10, verbose=False, ground="cells"):
    """Place a part mesh for every cell and every cell-to-cell link.

    ground says what generation 0 stands on: "cells" puts the square footing
    under each of them, "start" swaps them for the flat footed start cell, and
    "none" leaves them as ordinary cells with nothing underneath.
    """
    if ground not in ("cells", "start", "none"):
        raise ValueError(f"ground must be cells, start or none, not {ground!r}")
    plan = plan_structure(grid_record, verbose=verbose)
    meshes = []

    for (x, y, t) in plan["cells"]:
        part = "start" if (t == 0 and ground == "start") else "cell"
        m = mesh.Mesh(np.copy(blocks[part].data))  # mesh for each live cell
        m.x += x * unit
        m.y += y * unit
        m.z += t * unit
        meshes.append(mesh.Mesh(np.copy(m.data)))

        if t == 0 and ground == "cells":  # adding the base for printing
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
            "connection_lines": plan["connection_lines"], "unit": unit,
            "ground": ground}


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


# --------------------------------------------------- the generation 0 cell

RIM_SNAP = 4   # decimals; the part STLs are only clean to about this


def _rim_key(v):
    return (round(float(v[0]), RIM_SNAP), round(float(v[1]), RIM_SNAP),
            round(float(v[2]), RIM_SNAP))


def clip_above(vectors, z=0.0):
    """Keep the part of a triangle soup at or above a plane, splitting straddlers."""
    out = []
    for tri in vectors:
        poly = []
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            if a[2] >= z - 1e-9:
                poly.append(a)
            if (a[2] - z) * (b[2] - z) < 0:
                poly.append(a + (z - a[2]) / (b[2] - a[2]) * (b - a))
        if len(poly) >= 3:
            p = np.array(poly)
            for i in range(1, len(p) - 1):
                out.append([p[0], p[i], p[i + 1]])
    return np.array(out)


def rim_loop(tris, z=0.0):
    """The open boundary left by clipping, ordered into one loop."""
    count, store = {}, {}
    for t in tris:
        for i in range(3):
            a, b = _rim_key(t[i]), _rim_key(t[(i + 1) % 3])
            if a == b:
                continue
            e = tuple(sorted([a, b]))
            count[e] = count.get(e, 0) + 1
            store[e] = (a, b)
    rim = [store[e] for e, c in count.items() if c == 1]
    if not rim:
        raise ValueError("clipping left no open rim")
    stray = [e for e in rim if abs(e[0][2] - z) > 1e-3 or abs(e[1][2] - z) > 1e-3]
    if stray:
        raise ValueError(f"{len(stray)} open edges away from the cut plane; "
                         f"the part is not watertight enough to cut")
    nbr = {}
    for a, b in rim:
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    if any(len(v) != 2 for v in nbr.values()):
        raise ValueError("the cut boundary is not a simple loop")
    start = rim[0][0]
    loop, prev, cur = [start], None, start
    while True:
        a, b = tuple(nbr[cur])
        step = a if a != prev else b
        if step == start:
            break
        loop.append(step)
        prev, cur = cur, step
    if len(loop) != len(nbr):
        raise ValueError("the cut boundary has more than one loop")
    return np.array(loop, dtype=float)


def _skirted(top, loop, zbot, k, reverse):
    lp = loop[::-1] if reverse else loop
    n, cen = len(lp), np.array([0.0, 0.0, zbot])
    tris = list(top)
    for i in range(n):
        a, b = lp[i], lp[(i + 1) % n]
        a2 = np.array([a[0] * k, a[1] * k, zbot])
        b2 = np.array([b[0] * k, b[1] * k, zbot])
        tris += [[a, b, b2], [a, b2, a2], [cen, a2, b2]]
    data = np.zeros(len(tris), dtype=mesh.Mesh.dtype)
    data["vectors"] = np.array(tris)
    m = mesh.Mesh(data)
    m.update_normals()
    return m


def make_start_cell(cell, foot_radius=5.0):
    """A cell for generation 0 that can be printed without anything under it.

    The lower half of a cell narrows to a point, which cannot start a print.
    This keeps the upper half and replaces the lower half with a skirt flaring
    out to a flat foot on the build plate, the same idea as the cell_base part
    in model_stls/old.  The foot is one lattice pitch across by default, so
    neighbouring cells of generation 0 meet without overlapping, and the part
    occupies exactly the z range of an ordinary cell, so nothing else moves.
    """
    v = cell.vectors
    zbot = float(v.reshape(-1, 3)[:, 2].min())
    top = clip_above(v, 0.0)
    loop = rim_loop(top, 0.0)
    k = foot_radius / float(np.hypot(loop[:, 0], loop[:, 1]).max())
    import warnings
    m = _skirted(top, loop, zbot, k, reverse=False)
    with warnings.catch_warnings():
        # numpy-stl's closed check is a loose tolerance on the normal sum and
        # trips on this mesh; the rim pairing is verified in the test suite
        warnings.simplefilter("ignore")
        flipped = m.get_mass_properties()[0] < 0
    if flipped:                             # keep the normals pointing outwards
        m = _skirted(top, loop, zbot, k, reverse=True)
    return m


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

    A pattern that spreads as it runs is widest well above the plate, so the
    plate is sized to the whole model rather than to the few cells that touch it.
    """
    unit = model["unit"]
    pts = np.array([[x * unit, y * unit] for x, y, _ in model["cells"]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    cell_r = float(np.abs(blocks["cell"].points.reshape(-1, 3)[:, :2]).max())
    radius = float(np.linalg.norm(pts - centre, axis=1).max()) + cell_r
    return centre[0], centre[1], radius


def make_circular_base(model, blocks, radius=None, thickness=3.0, segments=180,
                       margin=2.0, weld_to=None):
    """Build the plate and return (mesh, centre_x, centre_y, radius).

    It spans from the underside of the footing part upwards, so it fuses with
    everything standing on the build plate.

    weld_to is a z the plate top must clear.  Pass it when no footings are
    placed: the plate then has to reach up into the generation 0 cells itself,
    and a plate thinner than the cells hang down would touch nothing at all.
    """
    cx, cy, auto_r = base_footprint(model, blocks)
    if radius is None:
        radius = auto_r + margin
    # -4mm: the underside of the footing part, which is where the build plate
    # is.  Used even when no footings are placed, since the generation 0 cells
    # reach down to -3.575 and so still meet a plate of any thickness.
    z0 = float(blocks["base"].points.reshape(-1, 3)[:, 2].min())
    if weld_to is not None and z0 + thickness < weld_to:
        raise ValueError(
            f"a {thickness}mm plate reaches z={z0 + thickness:.3f}, which is "
            f"below z={weld_to:.3f} where the generation 0 cells start, so it "
            f"would touch nothing. Use at least "
            f"{weld_to - z0:.2f}mm, or add footings with --base both.")

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
    src.add_argument("--array", help="pattern file of on/off cells "
                                     "(text, .npy or .npz); see init_grids.load_pattern")
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
    p.add_argument("--base", choices=["cells", "start", "none", "plate", "both"],
                   default="cells",
                   help="what goes underneath: 'cells' a square footing under "
                        "each cell of generation 0 (default); 'start' swaps "
                        "those cells for the flat footed start cell so the "
                        "model prints with no base at all; 'none' nothing, not "
                        "printable as it stands; 'plate' a round plate only; "
                        "'both' footings and plate")
    p.add_argument("--base-radius", type=float, default=None,
                   help="plate radius in mm (default: cover the whole model)")
    p.add_argument("--base-thickness", type=float, default=3.0,
                   help="plate thickness in mm")
    p.add_argument("--base-segments", type=int, default=180,
                   help="facets around the plate")
    p.add_argument("--boundary", choices=["grow", "wall"], default="grow",
                   help="'grow' lets the pattern spread past the edge of the "
                        "starting grid, which is true unbounded Life (default); "
                        "'wall' keeps the grid fixed and kills anything that "
                        "reaches the edge")
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
    elif args.array:
        grid = init_grids.load_pattern(args.array)
    else:
        grid = load_grid_from_image(args.image)

    if args.pad:
        grid = np.pad(grid, ((args.pad, args.pad), (args.pad, args.pad)))

    grid_record = life_run(grid, args.frames, boundary=args.boundary)
    blocks = load_building_blocks(args.model_stls)
    model = build_model(grid_record, blocks, unit=args.unit, verbose=args.verbose,
                        ground={"cells": "cells", "both": "cells",
                                "start": "start"}.get(args.base, "none"))

    conn = check_connectivity(model)

    out_meshes = list(model["meshes"])
    plate = None
    if args.base in ("plate", "both"):
        weld = None
        if args.base == "plate":    # no footings, the plate must reach the cells
            weld = float(blocks["cell"].points.reshape(-1, 3)[:, 2].min()) + 0.25
        try:
            plate, bx, by, br = make_circular_base(
                model, blocks, radius=args.base_radius,
                thickness=args.base_thickness, segments=args.base_segments,
                weld_to=weld)
        except ValueError as exc:
            p.error(str(exc))
        out_meshes.append(plate)

    if args.center:
        pts = np.concatenate([m.points.reshape(-1, 3) for m in out_meshes])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        shift = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])
        for m in out_meshes:
            m.x -= shift[0]
            m.y -= shift[1]

    combined = save_stl(out_meshes, args.out, ascii_mode=not args.binary)

    print(f"grid {grid.shape[0]}x{grid.shape[1]}, {args.frames} frames, "
          f"{args.boundary} boundary")
    if grid_record.shape[1:] != grid.shape:
        print(f"      grew to {grid_record.shape[1]}x{grid_record.shape[2]} so the "
              f"pattern was not cut off")
    elif args.boundary == "wall":
        # say whether the wall actually changed the outcome, not merely whether
        # the pattern came close to it
        free = life_run(grid, args.frames, boundary="grow")
        (r0, _), (c0, _) = grow_pad(grid, args.frames)
        window = free[:, r0:r0 + grid.shape[0], c0:c0 + grid.shape[1]]
        if free.sum() != window.sum() or not np.array_equal(window, grid_record):
            print(f"      warning: the wall cut this pattern off, losing "
                  f"{int(free.sum() - grid_record.sum())} cell parts. It is no "
                  f"longer Conway's Life. Use --boundary grow.")
    print(f"live cells: {len(model['cells'])}, rungs: {len(model['connection_lines'])}")
    print(f"parts: {len(out_meshes)}, triangles: {len(combined.data)}")
    print(f"connected pieces: {len(conn['components'])}"
          f" ({len(conn['floating'])} not reaching generation 0)")
    if plate is not None:
        print(f"base: round plate, radius {br:.1f}mm, {args.base_thickness}mm "
              f"thick, centred at ({bx:.1f}, {by:.1f})"
              + (" + footings" if args.base == "both" else ""))
    elif args.base == "cells":
        n = sum(1 for _, _, t in model["cells"] if t == 0)
        print(f"base: {n} square footings under generation 0")
    elif args.base == "start":
        n = sum(1 for _, _, t in model["cells"] if t == 0)
        print(f"base: none, generation 0 uses {n} flat footed start cells")
    else:
        print("base: none, the model is meant to be held rather than stood up")
        if len(conn["components"]) > 1:
            print(f"      note: {len(conn['components'])} loose pieces without "
                  f"a plate to join them")
    print(f"wrote {args.out}")

    if args.plot:
        plot_all(grid_record, model, blocks)

    return model


if __name__ == "__main__":
    main()
