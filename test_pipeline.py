"""End-to-end check of the Conway -> print-in-place pipeline.

Uses two gliders colliding into a pond as the test case.  Run directly
(``python test_pipeline.py``) or under pytest.
"""

import os
import tempfile

import numpy as np
from stl import mesh

import init_grids
from conway3d_stl import (base_footprint, build_model, check_connectivity,
                          conway_update, conwayog_rule_mask,
                          cylinder_mesh, life_run, load_building_blocks,
                          load_grid_from_image, make_circular_base, plan_structure,
                          save_stl)

class Skip(Exception):
    """Raised by a test that cannot run here, e.g. tkinter is not installed."""


MODEL_DIR = "./model_stls/version3"
UNIT = 10
FRAMES = 17  # collision settles into a pond at frame 15


# ------------------------------------------------------------------ life rules

def test_still_lifes_and_oscillators():
    """Sanity-check the rule implementation on known patterns."""
    block = np.zeros((6, 6), dtype=int)
    block[2:4, 2:4] = 1
    rec = life_run(block, 4)
    for t in range(4):
        assert np.array_equal(rec[t], block), f"block decayed at t={t}"

    blinker = np.zeros((6, 6), dtype=int)
    blinker[3, 2:5] = 1
    rec = life_run(blinker, 5)
    assert np.array_equal(rec[0], rec[2]), "blinker period is not 2"
    assert not np.array_equal(rec[0], rec[1]), "blinker did not oscillate"
    assert rec[1].sum() == 3, "blinker changed cell count"


def test_two_gliders_collide_into_a_pond():
    """The two gliders must actually meet and leave a pond behind."""
    grid = init_grids.two_glider
    rec = life_run(grid, 30)

    assert rec[0].sum() == 10, "two gliders should start as 10 live cells"

    # each glider alone is a 5-cell pattern that keeps its population
    for t in range(9):
        assert rec[t].sum() == 10, f"gliders lost cells before colliding (t={t})"

    # they interact: population departs from 10
    assert any(rec[t].sum() != 10 for t in range(10, 16)), "no collision happened"

    # and the result is a still life from frame 15 on
    for t in range(15, 29):
        assert np.array_equal(rec[t], rec[t + 1]), f"not settled at t={t}"

    final = rec[15]
    assert final.sum() == 8, f"expected an 8-cell pond, got {int(final.sum())}"
    ys, xs = np.nonzero(final)
    shape = final[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    pond = np.array([[0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [1, 0, 0, 1],
                     [0, 1, 1, 0]])
    assert np.array_equal(shape, pond), f"final still life is not a pond:\n{shape}"


def test_image_and_array_initial_conditions_agree():
    png = load_grid_from_image("initial_sates/twoGlider_pond.png")
    assert np.array_equal(png, init_grids.two_glider)


# -------------------------------------------------------------- part geometry

def test_part_stls_are_closed_surfaces():
    """Every building block must be watertight: each edge used by 2 triangles."""
    for name in ["base", "cell", "rung_adj", "rung_diag"]:
        m = mesh.Mesh.from_file(f"{MODEL_DIR}/{name}.stl")
        tris = m.vectors.round(4)
        edges = {}
        for tri in tris:
            keys = [tuple(v) for v in tri]
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                e = tuple(sorted([keys[a], keys[b]]))
                edges[e] = edges.get(e, 0) + 1
        bad = [c for c in edges.values() if c != 2]
        assert not bad, f"{name}.stl has {len(bad)} non-manifold edges"
        assert np.isfinite(m.vectors).all(), f"{name}.stl has non-finite vertices"


def test_blocks_are_oriented_towards_positive_x():
    blocks = load_building_blocks(MODEL_DIR)
    for key in ["rung_adj", "rung_crn"]:
        pts = blocks[key].points.reshape(-1, 3)
        assert pts[:, 0].max() > 0 and abs(pts[:, 1]).max() < 2, (
            f"{key} is not aimed along +x after orientation: "
            f"x {pts[:, 0].min():.2f}..{pts[:, 0].max():.2f}")

    # the adjacent rung must span one lattice step, the diagonal one sqrt(2)
    adj = blocks["rung_adj"].points.reshape(-1, 3)[:, 0].max()
    crn = blocks["rung_crn"].points.reshape(-1, 3)[:, 0].max()
    assert abs(adj - UNIT) < 1.0, f"rung_adj reaches {adj:.2f}, expected ~{UNIT}"
    assert abs(crn - UNIT * np.sqrt(2)) < 1.0, (
        f"rung_diag reaches {crn:.2f}, expected ~{UNIT * np.sqrt(2):.2f}")

    # the base must sit on the build plate
    base_z = blocks["base"].points.reshape(-1, 3)[:, 2]
    cell_z = blocks["cell"].points.reshape(-1, 3)[:, 2]
    assert abs(base_z.min() - -4) < 0.1, f"base does not start at -4mm: {base_z.min()}"
    assert base_z.max() >= cell_z.min(), "base does not reach up to the cell"


def test_rungs_bridge_the_cells_they_link():
    """Each placed rung must physically reach from its cell to the next one."""
    blocks = load_building_blocks(MODEL_DIR)
    cell_r = np.abs(blocks["cell"].points.reshape(-1, 3)).max()

    grid = np.zeros((7, 7), dtype=int)
    grid[2:5, 2:5] = 1  # a 3x3 block seeds links in all eight directions
    rec = life_run(grid, 3)
    model = build_model(rec, blocks, unit=UNIT)

    rung_meshes = []
    for m in model["meshes"]:
        n = len(m.data)
        if n in (len(blocks["rung_adj"].data), len(blocks["rung_crn"].data)):
            rung_meshes.append(m)
    assert rung_meshes, "no rungs were placed"
    assert len(rung_meshes) == len(model["connection_lines"]), (
        "rung mesh count does not match the recorded links")

    checked = 0
    for m, ([x0, y0, t0], [x1, y1, t1]) in zip(rung_meshes, model["connection_lines"]):
        src = np.array([x0 * UNIT, y0 * UNIT, t0 * UNIT])
        dst = np.array([x1 * UNIT, y1 * UNIT, t1 * UNIT])
        pts = m.points.reshape(-1, 3)

        # the rung must overlap the source cell at its foot and the destination
        # cell at its head, otherwise the print falls apart
        d_src = np.linalg.norm(pts - src, axis=1).min()
        d_dst = np.linalg.norm(pts - dst, axis=1).min()
        assert d_src < cell_r, (
            f"rung {src}->{dst} does not touch its source cell ({d_src:.2f}mm away)")
        assert d_dst < cell_r, (
            f"rung {src}->{dst} does not touch its destination cell "
            f"({d_dst:.2f}mm away)")

        # and it must point the right way, not at some other neighbour
        far = pts[np.argmax(np.linalg.norm(pts[:, :2] - src[:2], axis=1))]
        want = (dst - src)[:2]
        got = (far - src)[:2]
        cos = got @ want / (np.linalg.norm(got) * np.linalg.norm(want))
        assert cos > 0.97, (
            f"rung {src}->{dst} aims {np.degrees(np.arccos(cos)):.1f} deg off target")
        checked += 1

    assert checked >= 8, f"only checked {checked} rungs"


# --------------------------------------------------------------- full pipeline

def test_two_glider_model_is_one_printable_piece():
    blocks = load_building_blocks(MODEL_DIR)
    rec = life_run(init_grids.two_glider, FRAMES)
    model = build_model(rec, blocks, unit=UNIT)

    assert len(model["cells"]) == int(rec.sum()), "cell count != live cells"
    assert model["connection_lines"], "no rungs placed"

    conn = check_connectivity(model)
    assert not conn["floating"], (
        f"{len(conn['floating'])} piece(s) never reach the base: "
        f"{[len(g) for g in conn['floating']]}")
    assert len(conn["components"]) == 1, (
        f"model is in {len(conn['components'])} pieces, sizes "
        f"{[len(g) for g in conn['components']]}")


def test_written_stl_round_trips():
    blocks = load_building_blocks(MODEL_DIR)
    rec = life_run(init_grids.two_glider, FRAMES)
    model = build_model(rec, blocks, unit=UNIT)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "two_glider_pond.stl")
        combined = save_stl(model["meshes"], path, ascii_mode=False)
        assert os.path.getsize(path) > 0

        back = mesh.Mesh.from_file(path)
        assert len(back.data) == len(combined.data), "triangle count changed on save"
        assert len(back.data) == sum(len(m.data) for m in model["meshes"])
        assert np.isfinite(back.vectors).all(), "STL contains non-finite vertices"
        assert np.allclose(back.vectors, combined.vectors, atol=1e-3)

        pts = back.points.reshape(-1, 3)
        assert pts[:, 2].min() < 0.1, "model does not sit on the build plate"
        expect_top = (FRAMES - 1) * UNIT
        assert abs(pts[:, 2].max() - expect_top) < 5, (
            f"top of model at {pts[:, 2].max():.1f}mm, expected ~{expect_top}mm")


# ---------------------------------------------------------------- regression

def test_matches_original_project_output():
    """Lock the refactor to the model the original project shipped.

    model_stls/version1 with 16 frames is what produced twoGlider_toPond.stl in
    the pre-repo working directory.  Its geometry is reproduced exactly here,
    up to a one-lattice-unit translation in x (the old run padded the grid
    differently), so these numbers pin the placement logic.
    """
    blocks = load_building_blocks("./model_stls/version1")
    rec = life_run(init_grids.two_glider, 16)
    model = build_model(rec, blocks, unit=UNIT)

    tris = sum(len(m.data) for m in model["meshes"])
    assert tris == 456260, f"triangle count drifted: {tris} != 456260"

    pts = np.concatenate([m.points.reshape(-1, 3) for m in model["meshes"]])
    size = pts.max(axis=0) - pts.min(axis=0)
    assert np.allclose(size, [120.00, 67.97, 156.98], atol=0.01), (
        f"bounding box drifted: {np.round(size, 2)}")

    assert len(model["cells"]) == 145, f"cell count drifted: {len(model['cells'])}"
    assert len(model["connection_lines"]) == 367, (
        f"rung count drifted: {len(model['connection_lines'])}")


def test_every_part_version_builds_a_printable_model():
    """All three shipped part sets must place and link correctly."""
    for version in ["version1", "version2", "version3"]:
        blocks = load_building_blocks(f"./model_stls/{version}")
        rec = life_run(init_grids.two_glider, FRAMES)
        model = build_model(rec, blocks, unit=UNIT)
        conn = check_connectivity(model)
        assert len(conn["components"]) == 1, (
            f"{version}: model is in {len(conn['components'])} pieces")
        assert not conn["floating"], f"{version}: pieces do not reach the base"

        cell_r = np.abs(blocks["cell"].points.reshape(-1, 3)).max()
        rung_sizes = {len(blocks["rung_adj"].data), len(blocks["rung_crn"].data)}
        rungs = [m for m in model["meshes"] if len(m.data) in rung_sizes]
        assert len(rungs) == len(model["connection_lines"]), (
            f"{version}: rung count mismatch")
        for m, ([x0, y0, t0], [x1, y1, t1]) in zip(rungs, model["connection_lines"]):
            src = np.array([x0 * UNIT, y0 * UNIT, t0 * UNIT])
            dst = np.array([x1 * UNIT, y1 * UNIT, t1 * UNIT])
            pts = m.points.reshape(-1, 3)
            assert np.linalg.norm(pts - src, axis=1).min() < cell_r, (
                f"{version}: rung {src}->{dst} misses its source cell")
            assert np.linalg.norm(pts - dst, axis=1).min() < cell_r, (
                f"{version}: rung {src}->{dst} misses its destination cell")


# -------------------------------------------------------------- circular base

def test_cylinder_is_a_closed_solid_with_outward_normals():
    m = cylinder_mesh(0, 0, -4, -1, 50, segments=180)
    vol, cog, _ = m.get_mass_properties()
    assert vol > 0, "normals point inward"
    assert abs(vol - np.pi * 50 ** 2 * 3) / vol < 0.01, f"volume off: {vol}"
    assert np.allclose(cog, [0, 0, -2.5], atol=1e-6), f"centre of gravity {cog}"

    tris = m.vectors.round(4)
    edges = {}
    for tri in tris:
        k = [tuple(v) for v in tri]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            e = tuple(sorted([k[a], k[b]]))
            edges[e] = edges.get(e, 0) + 1
    assert all(c == 2 for c in edges.values()), "plate is not watertight"


def test_plate_supports_the_whole_model():
    """The plate must catch every cell that touches it and out to the canopy."""
    blocks = load_building_blocks(MODEL_DIR)
    rec = life_run(init_grids.two_glider, FRAMES)
    model = build_model(rec, blocks, unit=UNIT)

    plate, cx, cy, r = make_circular_base(model, blocks, thickness=3.0)
    unit = model["unit"]

    for x, y, t in model["cells"]:
        d = np.hypot(x * unit - cx, y * unit - cy)
        assert d <= r, f"cell {(x, y, t)} overhangs the plate by {d - r:.1f}mm"

    # it must sit on the build plate and fuse with the per-cell bases
    pz = plate.points.reshape(-1, 3)[:, 2]
    base_z = blocks["base"].points.reshape(-1, 3)[:, 2]
    assert abs(pz.min() - base_z.min()) < 1e-6, "plate does not sit at z of the bases"
    assert pz.max() > base_z.min(), "plate has no thickness"
    assert pz.max() <= base_z.max() + 1e-9, "plate swallows the cells above it"


def test_explicit_base_radius_is_honoured():
    blocks = load_building_blocks(MODEL_DIR)
    model = build_model(life_run(init_grids.two_glider, 6), blocks, unit=UNIT)
    plate, cx, cy, r = make_circular_base(model, blocks, radius=40.0)
    assert abs(r - 40.0) < 1e-9
    pts = plate.points.reshape(-1, 3)
    reach = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).max()
    assert abs(reach - 40.0) < 0.01, f"plate reaches {reach:.2f}, asked for 40"


# ----------------------------------------------------------- structure vs mesh

def test_plan_matches_the_placed_meshes():
    """plan_structure and build_model must agree, since preview.py uses the plan."""
    blocks = load_building_blocks(MODEL_DIR)
    rec = life_run(init_grids.two_glider, FRAMES)
    plan = plan_structure(rec)
    model = build_model(rec, blocks, unit=UNIT)

    assert plan["cells"] == model["cells"]
    assert plan["connection_lines"] == model["connection_lines"]
    assert len(plan["rungs"]) == len(plan["connection_lines"])

    n_base = sum(1 for _, _, t in plan["cells"] if t == 0)
    expect = len(plan["cells"]) + n_base + len(plan["rungs"])
    assert len(model["meshes"]) == expect, (
        f"{len(model['meshes'])} meshes placed, plan implies {expect}")


# --------------------------------------------------------------------- trees

TREE_EXPECT = {          # cells, rungs at 22 frames, from tree_search.py
    "tree_fork":     (210, 539),
    "tree_cross":    (257, 672),
    "tree_slender":  (239, 609),
    "tree_crown":    (252, 628),
    "tree_pine":     (609, 1636),
}
TREE_FRAMES = 22


def test_tree_patterns_have_the_expected_shape():
    for name, (n_cells, n_rungs) in TREE_EXPECT.items():
        grid = getattr(init_grids, name)
        rec = life_run(grid, TREE_FRAMES)
        plan = plan_structure(rec)
        assert len(plan["cells"]) == n_cells, (
            f"{name}: {len(plan['cells'])} cells, expected {n_cells}")
        assert len(plan["rungs"]) == n_rungs, (
            f"{name}: {len(plan['rungs'])} rungs, expected {n_rungs}")


def test_trees_never_touch_the_grid_boundary():
    """A pattern that reaches the edge is clipped, and the shape is a lie."""
    for name in init_grids.TREE_ART:
        rec = life_run(getattr(init_grids, name), TREE_FRAMES)
        for side in (rec[:, :2, :], rec[:, -2:, :], rec[:, :, :2], rec[:, :, -2:]):
            assert not side.any(), f"{name} reaches the edge of its grid"


def test_trees_grow_upward_and_outward():
    """A tree stands on a small foot and opens out above it."""
    for name in init_grids.TREE_ART:
        rec = life_run(getattr(init_grids, name), TREE_FRAMES)
        radius = []
        for fr in rec:
            ys, xs = np.nonzero(fr)
            c = np.array([ys.mean(), xs.mean()])
            radius.append(np.linalg.norm(np.c_[ys, xs] - c, axis=1).max())
        radius = np.array(radius)
        f = len(radius)

        assert rec[-1].sum() > 0, f"{name} dies out before the top"
        # the part that touches the plate must be the small end
        assert radius[0] <= radius[2 * f // 3:].mean(), (
            f"{name} starts wider than it ends, so it is standing on its crown")
        assert radius[2 * f // 3:].mean() > radius[:f // 3].mean(), (
            f"{name} does not widen towards the top")


def test_trees_print_as_one_piece_on_a_plate_that_holds_them_up():
    blocks = load_building_blocks(MODEL_DIR)
    for name in init_grids.TREE_ART:
        rec = life_run(getattr(init_grids, name), TREE_FRAMES)
        model = build_model(rec, blocks, unit=UNIT)

        conn = check_connectivity(model)
        assert len(conn["components"]) == 1, (
            f"{name}: {len(conn['components'])} separate pieces")
        assert not conn["floating"], f"{name}: a piece never reaches the plate"

        _, cx, cy, r = make_circular_base(model, blocks)
        pts = np.array([[x * UNIT, y * UNIT] for x, y, _ in model["cells"]])
        assert np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).max() <= r, (
            f"{name} overhangs its plate")

        # a top heavy tree must keep its centre of mass well inside the plate,
        # every cell being the same part and so the same weight
        com = pts.mean(axis=0)
        lean = float(np.hypot(com[0] - cx, com[1] - cy))
        assert lean < 0.5 * r, (
            f"{name} leans {lean:.0f}mm off a {r:.0f}mm plate, it would tip")


def test_from_art_round_trips():
    grid = init_grids.from_art(["#.#", ".#.", "#.#"], size=11)
    assert grid.shape == (11, 11)
    assert grid.sum() == 5
    ys, xs = np.nonzero(grid)
    sub = grid[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    assert np.array_equal(sub, [[1, 0, 1], [0, 1, 0], [1, 0, 1]])


# ------------------------------------------------- the rules really are Life

def _art(rows):
    return np.array([[1 if c == "#" else 0 for c in r] for r in rows], dtype=int)


def test_rule_is_exactly_b3_s23():
    """Every one of the 18 (state, neighbour count) cases, not a sample."""
    for state in (0, 1):
        for n in range(9):
            want = 1 if (n == 3 or (state == 1 and n == 2)) else 0
            got = conway_update(state, n)
            assert got == want, (
                f"state {state} with {n} neighbours gave {got}, Life says {want}")

    assert conwayog_rule_mask.sum() == 8, "neighbourhood is not the 8 around a cell"
    assert conwayog_rule_mask[1, 1] == 0, "a cell is counted as its own neighbour"
    assert conwayog_rule_mask.shape == (3, 3)


CANONICAL = {
    # name: (seed, period, displacement per period)
    "block":   (_art(["##", "##"]), 1, (0, 0)),
    "beehive": (_art([".##.", "#..#", ".##."]), 1, (0, 0)),
    "loaf":    (_art([".##.", "#..#", ".#.#", "..#."]), 1, (0, 0)),
    "boat":    (_art(["##.", "#.#", ".#."]), 1, (0, 0)),
    "tub":     (_art([".#.", "#.#", ".#."]), 1, (0, 0)),
    "blinker": (_art(["###"]), 2, (0, 0)),
    "toad":    (_art([".###", "###."]), 2, (0, 0)),
    "beacon":  (_art(["##..", "##..", "..##", "..##"]), 2, (0, 0)),
    "pulsar":  (_art(["..###...###..", ".............",
                      "#....#.#....#", "#....#.#....#", "#....#.#....#",
                      "..###...###..", ".............", "..###...###..",
                      "#....#.#....#", "#....#.#....#", "#....#.#....#",
                      ".............", "..###...###.."]), 3, (0, 0)),
    "glider":  (_art([".#.", "..#", "###"]), 4, (1, 1)),
    "lwss":    (_art([".####", "#...#", "....#", "#..#."]), 4, (0, 2)),
}


def test_canonical_patterns_have_their_known_periods():
    """Still lifes hold, oscillators return, spaceships move by the right step."""
    for name, (seed, period, disp) in CANONICAL.items():
        rec = life_run(np.pad(seed, 20), period * 3 + 1)
        start = rec[0]

        for k in (1, 2, 3):
            want = np.roll(np.roll(start, disp[0] * k, axis=0), disp[1] * k, axis=1)
            assert np.array_equal(rec[period * k], want), (
                f"{name} is wrong after {period * k} generations")

        for shorter in range(1, period):
            assert not np.array_equal(rec[shorter], start), (
                f"{name} repeats at {shorter}, so its period is not {period}")


def test_r_pentomino_matches_the_documented_result():
    """The R-pentomino settles at generation 1103 with 116 cells.

    A long, well documented run catches rule errors too subtle to show up in
    small patterns.  The field is wide enough that the six escaping gliders
    never reach the edge, so this is infinite-plane Life.
    """
    from tree_search import step   # verified against life_run below

    n = 641
    g = np.zeros((1, n, n), dtype=np.int8)
    c = n // 2
    for r, x in [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]:
        g[0, c + r, c + x] = 1

    for _ in range(1103):
        g = step(g)
    assert int(g.sum()) == 116, f"generation 1103 has {int(g.sum())} cells, not 116"

    for _ in range(60):                      # and it stays settled
        g = step(g)
        assert int(g.sum()) == 116, "population moved after it should have settled"

    assert not (g[0, :2, :].any() or g[0, -2:, :].any()
                or g[0, :, :2].any() or g[0, :, -2:].any()), "hit the boundary"


def test_fast_simulator_matches_the_pipeline():
    from tree_search import run_batch
    for name in ["tree_fork", "tree_pine"]:
        g = getattr(init_grids, name)
        a = life_run(g, 22).astype(int)
        b = run_batch(np.array([g], dtype=np.int8), 22)[0].astype(int)
        assert np.array_equal(a, b), f"{name}: the two simulators disagree"


def test_trees_are_unaffected_by_the_size_of_their_grid():
    """A pattern touching the edge would be silently clipped and not be Life."""
    def shapes(rec):
        out = []
        for fr in rec:
            ys, xs = np.nonzero(fr)
            out.append(tuple(sorted(zip(ys - ys.min(), xs - xs.min()))))
        return out

    for name, art in init_grids.TREE_ART.items():
        small = life_run(init_grids.from_art(art, size=49), TREE_FRAMES)
        big = life_run(init_grids.from_art(art, size=141), TREE_FRAMES)
        assert shapes(small) == shapes(big), (
            f"{name} evolves differently on a bigger grid, so it is being clipped")


def test_placed_cells_reproduce_the_simulated_history():
    """The print is a faithful record: one cell part per live cell, nothing else."""
    for name in init_grids.TREE_ART:
        rec = life_run(getattr(init_grids, name), TREE_FRAMES)
        plan = plan_structure(rec)

        h, w = rec.shape[1] + 2, rec.shape[2] + 2
        back = np.zeros((rec.shape[0], h, w), dtype=int)
        for x, y, t in plan["cells"]:
            back[t, x, y] = 1
        back = back[:, ::-1, :][:, 1:-1, 1:-1]   # undo the flip and the padding

        assert np.array_equal(back, rec.astype(int)), (
            f"{name}: placed cells do not match the simulation")


# ------------------------------------------------------- pattern files + GUI

def test_pattern_text_formats_all_parse():
    glider = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]
    for label, text in [
        ("ascii art",      ".#.\n..#\n###"),
        ("plaintext O/.",  ".O.\n..O\nOOO"),
        ("comma 0/1",      "0,1,0\n0,0,1\n1,1,1"),
        ("space 0/1",      "0 1 0\n0 0 1\n1 1 1"),
        ("tab 0/1",        "0\t1\t0\n0\t0\t1\n1\t1\t1"),
        ("with comments",  "! a glider\n.#.\n..#\n###"),
        ("asterisks",      ".*.\n..*\n***"),
    ]:
        got = init_grids.parse_art(text)
        assert np.array_equal(got, glider), f"{label} parsed as {got.tolist()}"


def test_ragged_rows_are_padded_with_dead_cells():
    g = init_grids.parse_art("#\n###\n#")
    assert g.shape == (3, 3)
    assert np.array_equal(g, [[1, 0, 0], [1, 1, 1], [1, 0, 0]])


def test_pattern_file_round_trip_preserves_the_domain():
    """Blank rows are part of the domain the user chose, so they must survive."""
    import tempfile
    for name in ["tree_fork", "tree_pine", "two_glider"]:
        grid = getattr(init_grids, name)
        with tempfile.TemporaryDirectory() as d:
            for ext in (".txt", ".npy"):
                p = os.path.join(d, "p" + ext)
                init_grids.save_pattern(grid, p, comment="round trip")
                back = init_grids.load_pattern(p)
                assert back.shape == grid.shape, (
                    f"{name}{ext}: domain changed from {grid.shape} to {back.shape}")
                assert np.array_equal(back, grid), f"{name}{ext}: cells changed"


def test_pattern_loader_rejects_junk():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.txt")
        with open(p, "w") as fh:
            fh.write("#.#\n#Q#\n")
        try:
            init_grids.load_pattern(p)
        except ValueError as e:
            assert "Q" in str(e)
        else:
            raise AssertionError("an unknown character should not be accepted")

        empty = os.path.join(d, "empty.txt")
        open(empty, "w").close()
        try:
            init_grids.load_pattern(empty)
        except ValueError:
            pass
        else:
            raise AssertionError("an empty file should not be accepted")


def test_array_input_gives_the_same_model_as_the_named_pattern():
    import tempfile
    from conway3d_stl import main as build_main
    with tempfile.TemporaryDirectory() as d:
        pat = os.path.join(d, "fork.txt")
        init_grids.save_pattern(init_grids.tree_fork, pat)
        a = build_main(["--array", pat, "--frames", "8", "--binary",
                        "--out", os.path.join(d, "a.stl")])
        b = build_main(["--pattern", "tree_fork", "--frames", "8", "--binary",
                        "--out", os.path.join(d, "b.stl")])
    assert a["cells"] == b["cells"]
    assert a["connection_lines"] == b["connection_lines"]


def _designer():
    """designer.py needs tkinter, which is a separate package on some distros."""
    try:
        import designer
    except ImportError as exc:                 # pragma: no cover
        raise Skip(f"designer unavailable: {exc}")
    return designer


def test_resize_grid_keeps_the_drawing_centred():
    resize_grid = _designer().resize_grid
    g = init_grids.from_art(["###", "#.#", "###"], size=9)
    assert int(g.sum()) == 8

    grown = resize_grid(g, 21, 21)
    assert grown.shape == (21, 21)
    assert int(grown.sum()) == 8, "growing the domain lost cells"

    ys, xs = np.nonzero(grown)
    assert abs((ys.min() + ys.max()) / 2 - 10) <= 1, "not centred after growing"
    assert abs((xs.min() + xs.max()) / 2 - 10) <= 1, "not centred after growing"

    # shrinking crops rather than erroring
    cropped = resize_grid(g, 3, 3)
    assert cropped.shape == (3, 3)


def test_designer_analysis_warns_about_a_clipped_domain():
    analyse = _designer().analyse
    roomy = analyse(init_grids.tree_pine, TREE_FRAMES)
    assert not roomy["clipped"], "49x49 should be roomy enough for tree_pine"
    assert roomy["pieces"] == 1
    assert roomy["cells"] == TREE_EXPECT["tree_pine"][0]

    cramped = analyse(init_grids.from_art(init_grids.TREE_ART["tree_pine"], 21),
                      TREE_FRAMES)
    assert cramped["clipped"], "a 21x21 domain must be reported as clipping"

    assert analyse(np.zeros((10, 10), dtype=int), 5)["empty"]


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = skipped = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Skip as e:
            skipped += 1
            print(f"SKIP  {t.__name__}  ({e})")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}\n      {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}\n      {type(e).__name__}: {e}")
    ran = len(tests) - skipped
    tail = f", {skipped} skipped" if skipped else ""
    print(f"\n{ran - failed}/{ran} passed{tail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
