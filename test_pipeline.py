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
                          conway_update, conwayog_rule_mask, grow_pad,
                          room_to_grow,
                          cylinder_mesh, life_run, load_building_blocks,
                          load_grid_from_image, make_circular_base, plan_structure,
                          save_stl)

# A batched Life step, kept here because the long documented runs above need
# thousands of generations and the pipeline's own life_run is too slow for that.
# test_fast_simulator_matches_the_pipeline checks the two agree.
def step(g):
    """One generation of a stack of grids, zero boundary, matching life_run."""
    p = np.pad(g, ((0, 0), (1, 1), (1, 1)))
    n = (p[:, :-2, :-2] + p[:, :-2, 1:-1] + p[:, :-2, 2:] +
         p[:, 1:-1, :-2] + p[:, 1:-1, 2:] +
         p[:, 2:, :-2] + p[:, 2:, 1:-1] + p[:, 2:, 2:])
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.int8)


def run_batch(seeds, frames):
    """Evolve a stack of grids, returning (batch, frames, h, w)."""
    hist = np.empty((seeds.shape[0], frames, *seeds.shape[1:]), dtype=np.int8)
    hist[:, 0] = seeds
    g = seeds
    for f in range(1, frames):
        g = step(g)
        hist[:, f] = g
    return hist


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
    rec = life_run(block, 4, boundary="wall")   # fixed grid, this is a rule test
    for t in range(4):
        assert np.array_equal(rec[t], block), f"block decayed at t={t}"

    blinker = np.zeros((6, 6), dtype=int)
    blinker[3, 2:5] = 1
    rec = life_run(blinker, 5, boundary="wall")
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
    png = load_grid_from_image("initial_states/twoGlider_pond.png")
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


# ----------------------------------------------------- the named Life patterns
#
# init_grids.LIFE_ART is the single definition of these shapes; the CLI reads
# the same dict, so a pattern that is wrong here is wrong in the models too.

PATTERN_FRAMES = {"gosper_glider_gun": 40, "pentadecathlon": 31}
# a vanishing pattern is run exactly as long as it lives
VANISH_FRAMES = {"snowflake": 9, "fuse_diagonal": 10, "pinwheel_web": 11,
                 "row_of_six": 12, "woven_square": 13, "walled_box": 14,
                 "ring_of_eight": 15}
DEFAULT_FRAMES = 22


def frames_for(name):
    if name in VANISH_FRAMES:
        return VANISH_FRAMES[name]
    return PATTERN_FRAMES.get(name, DEFAULT_FRAMES)


# Period, and how far the object moves in one period.  Documented values.
KNOWN_BEHAVIOUR = {
    "block": (1, (0, 0)), "beehive": (1, (0, 0)), "loaf": (1, (0, 0)),
    "boat": (1, (0, 0)), "tub": (1, (0, 0)),
    "blinker": (2, (0, 0)), "toad": (2, (0, 0)), "beacon": (2, (0, 0)),
    "pulsar": (3, (0, 0)), "pentadecathlon": (15, (0, 0)),
    "glider": (4, (1, 1)), "lwss": (4, (0, 2)),
    "mwss": (4, (0, 2)), "hwss": (4, (0, 2)),
    # longer than four, so the print repeats only every p layers
    "octagon2": (5, (0, 0)), "figure_eight": (8, (0, 0)),
    "kok_galaxy": (8, (0, 0)), "tumbler": (14, (0, 0)),
}

# Patterns that build up and then vanish completely: the generation at which
# the last cell dies.  These are the closed forms, beginning and ending at
# nothing, so the number is what sets the height of the print.
KNOWN_VANISHING = {
    "snowflake": 9, "fuse_diagonal": 10, "pinwheel_web": 11, "row_of_six": 12,
    "woven_square": 13, "walled_box": 14, "ring_of_eight": 15,
}

# Generation at which the pattern stops changing, and the population there.
KNOWN_LIFESPAN = {
    "r_pentomino": (1103, 116),
    "b_heptomino": (148, 28),
    "pi_heptomino": (173, 55),
    "diehard": (130, 0),
}


def test_every_named_pattern_is_reachable_from_the_command_line():
    """--pattern does getattr on the module, so each name must be bound."""
    for name in init_grids.LIFE_ART:
        grid = getattr(init_grids, name, None)
        assert grid is not None, f"{name} is in LIFE_ART but not defined as a grid"
        assert grid.ndim == 2 and grid.sum() > 0, f"{name} is empty"


def test_named_patterns_have_their_known_periods():
    """Still lifes hold, oscillators return, spaceships move by the right step."""
    for name, (period, disp) in KNOWN_BEHAVIOUR.items():
        seed = np.pad(init_grids.from_art(init_grids.LIFE_ART[name]), 20)
        rec = life_run(seed, period * 3 + 1, boundary="wall")
        start = rec[0]

        for k in (1, 2, 3):
            shifted = [np.roll(np.roll(start, dy * k, 0), dx * k, 1)
                       for dy in (disp[0], -disp[0]) for dx in (disp[1], -disp[1])]
            assert any(np.array_equal(rec[period * k], w) for w in shifted), (
                f"{name} is wrong after {period * k} generations")

        for shorter in range(1, period):
            assert not np.array_equal(rec[shorter], start), (
                f"{name} repeats at {shorter}, so its period is not {period}")


def test_methuselahs_match_their_documented_lifespans():
    """A long run is the sharpest test of the rule: an error compounds."""
    for name, (gen, pop) in KNOWN_LIFESPAN.items():
        n = 801 if gen > 400 else 401
        g = np.zeros((1, n, n), dtype=np.int8)
        art = init_grids.from_art(init_grids.LIFE_ART[name], size=n)
        g[0] = art

        history = []
        for _ in range(gen + 60):
            g = step(g)
            history.append(int(g.sum()))

        assert history[gen - 1] == pop, (
            f"{name} has {history[gen - 1]} cells at generation {gen}, "
            f"documented is {pop}")
        assert len(set(history[gen - 1:])) == 1, (
            f"{name} is still changing after generation {gen}")


def test_gosper_gun_emits_one_glider_every_thirty_generations():
    rec = life_run(init_grids.gosper_glider_gun, 121)
    pop = [int(f.sum()) for f in rec]
    for t in (0, 30, 60, 90):
        assert pop[t + 30] - pop[t] == 5, (
            f"generations {t}-{t + 30} gained {pop[t + 30] - pop[t]} cells, "
            "a gun should gain exactly one 5-cell glider")


def test_named_patterns_are_not_clipped_by_their_grid():
    """A pattern touching the edge would be silently cut off and not be Life."""
    def shapes(rec):
        out = []
        for fr in rec:
            ys, xs = np.nonzero(fr)
            if len(ys) == 0:
                out.append(())
                continue
            out.append(tuple(sorted(zip(ys - ys.min(), xs - xs.min()))))
        return out

    for name, art in init_grids.LIFE_ART.items():
        n = frames_for(name)
        size = init_grids.LIFE_SIZE.get(name, init_grids.DEFAULT_GRID_SIZE)
        small = life_run(init_grids.from_art(art, size=size), n)
        big = life_run(init_grids.from_art(art, size=size + 80), n)
        assert shapes(small) == shapes(big), (
            f"{name} evolves differently on a bigger grid, so it is being clipped")


def test_named_patterns_print_as_a_plate_can_hold():
    """Either one solid, or several that a plate joins.  Nothing may float."""
    blocks = load_building_blocks(MODEL_DIR)
    for name in init_grids.LIFE_ART:
        rec = life_run(getattr(init_grids, name), frames_for(name))
        conn = check_connectivity(build_model(rec, blocks, unit=UNIT))
        assert not conn["floating"], (
            f"{name}: {len(conn['floating'])} pieces never reach the plate, "
            "so no base can hold them")


def test_placed_cells_reproduce_the_simulated_history():
    """The print is a faithful record: one cell part per live cell, nothing else."""
    for name in ["glider", "r_pentomino", "pulsar", "gosper_glider_gun"]:
        rec = life_run(getattr(init_grids, name), frames_for(name))
        plan = plan_structure(rec)

        h, w = rec.shape[1] + 2, rec.shape[2] + 2
        back = np.zeros((rec.shape[0], h, w), dtype=int)
        for x, y, t in plan["cells"]:
            back[t, x, y] = 1
        back = back[:, ::-1, :][:, 1:-1, 1:-1]   # undo the flip and the padding

        assert np.array_equal(back, rec.astype(int)), (
            f"{name}: placed cells do not match the simulation")


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


def test_fast_simulator_matches_the_pipeline():
    """`step` is used above for the long runs, so it has to agree with life_run."""
    for name in ["r_pentomino", "pulsar", "glider", "gosper_glider_gun"]:
        g = getattr(init_grids, name)
        n = frames_for(name)
        # run_batch is a fixed-grid, zero-boundary simulator, which is what
        # boundary="wall" is; "grow" would pad and the shapes would not line up
        a = life_run(g, n, boundary="wall").astype(int)
        b = run_batch(np.array([g], dtype=np.int8), n)[0].astype(int)
        assert np.array_equal(a, b), f"{name}: the two simulators disagree"


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
    for name in ["glider", "pi_heptomino", "two_glider"]:
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
        pat = os.path.join(d, "seed.txt")
        init_grids.save_pattern(init_grids.r_pentomino, pat)
        a = build_main(["--array", pat, "--frames", "8", "--binary",
                        "--out", os.path.join(d, "a.stl")])
        b = build_main(["--pattern", "r_pentomino", "--frames", "8", "--binary",
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
    PI_CELLS = 609   # pi heptomino at 22 generations

    roomy = analyse(init_grids.pi_heptomino, DEFAULT_FRAMES)
    assert not roomy["clipped"], "the stock grid should be roomy enough"
    assert roomy["pieces"] == 1
    assert roomy["cells"] == PI_CELLS

    # 21x21 looks tight but the wall never actually bites, and saying otherwise
    # would be a false alarm
    roomy_enough = init_grids.from_art(init_grids.LIFE_ART["pi_heptomino"], 21)
    assert not analyse(roomy_enough, DEFAULT_FRAMES, boundary="wall")["clipped"]
    assert analyse(roomy_enough, DEFAULT_FRAMES, boundary="wall")["cells"] == PI_CELLS

    cramped = init_grids.from_art(init_grids.LIFE_ART["pi_heptomino"], 15)
    walled = analyse(cramped, DEFAULT_FRAMES, boundary="wall")
    assert walled["clipped"], "a 15x15 wall does cut the pattern off"
    assert walled["lost"] > 0, "cells were lost but none were reported"
    assert walled["cells"] < PI_CELLS

    # the same drawing with a growing boundary is not clipped, it is enlarged
    grown = analyse(cramped, DEFAULT_FRAMES, boundary="grow")
    assert not grown["clipped"], "nothing is cut off when the grid can grow"
    assert grown["grew"] != (0, 0), "the grid should have had to grow"
    assert grown["cells"] == PI_CELLS, (
        "growing from a cramped grid must give the same model as a roomy one")

    assert analyse(np.zeros((10, 10), dtype=int), 5)["empty"]


# ---------------------------------------------------------------- base modes

def _parts(ground):
    blocks = load_building_blocks(MODEL_DIR)
    rec = life_run(init_grids.r_pentomino, 10)
    return blocks, rec, build_model(rec, blocks, unit=UNIT, ground=ground)


def test_base_modes_place_exactly_the_right_parts():
    blocks, rec, with_feet = _parts("cells")
    _, _, without = _parts("none")

    n_cells = len(with_feet["cells"])
    n_rungs = len(with_feet["connection_lines"])
    n_feet = sum(1 for _, _, t in with_feet["cells"] if t == 0)
    assert n_feet > 0, "the test pattern should touch the ground"

    assert len(without["meshes"]) == n_cells + n_rungs, (
        "a model with no footings must be cells and rungs and nothing else")
    assert len(with_feet["meshes"]) == n_cells + n_rungs + n_feet, (
        "one footing per generation 0 cell")
    assert without["cells"] == with_feet["cells"], "footings changed the cells"
    assert without["connection_lines"] == with_feet["connection_lines"]


def test_handheld_model_has_nothing_below_the_cells():
    """With no base at all the lowest point is the underside of a cell."""
    blocks, _, model = _parts("none")
    pts = np.concatenate([m.points.reshape(-1, 3) for m in model["meshes"]])
    cell_bottom = float(blocks["cell"].points.reshape(-1, 3)[:, 2].min())
    assert abs(pts[:, 2].min() - cell_bottom) < 1e-6, (
        f"lowest point is {pts[:, 2].min():.3f}, expected the cell at {cell_bottom:.3f}")

    # and it is still one piece, since footings were never what held it together
    assert len(check_connectivity(model)["components"]) == 1


def test_start_cell_is_a_closed_solid_that_prints_flat():
    """The flat footed cell that replaces generation 0 when there is no base."""
    for version in ["version1", "version2", "version3"]:
        blocks = load_building_blocks(f"./model_stls/{version}")
        start, cell = blocks["start"], blocks["cell"]
        sp = start.points.reshape(-1, 3)
        cp = cell.points.reshape(-1, 3)

        # it must occupy exactly the z range of an ordinary cell, so that
        # swapping it in moves nothing else
        assert abs(sp[:, 2].min() - cp[:, 2].min()) < 1e-6, f"{version}: bottom moved"
        assert abs(sp[:, 2].max() - cp[:, 2].max()) < 1e-6, f"{version}: top moved"

        # a flat foot exactly one lattice pitch across, so neighbours meet
        foot = sp[np.isclose(sp[:, 2], sp[:, 2].min(), atol=1e-6)]
        assert len(foot) > 8, f"{version}: no flat face at the bottom"
        across = 2 * np.hypot(foot[:, 0], foot[:, 1]).max()
        assert abs(across - UNIT) < 0.01, (
            f"{version}: foot is {across:.2f}mm across, expected {UNIT}")

        # closed, and every edge traversed once each way
        seen = {}
        for t in start.vectors:
            for i in range(3):
                a = tuple(np.round(t[i], 4))
                b = tuple(np.round(t[(i + 1) % 3], 4))
                if (b, a) in seen:
                    seen[(b, a)] -= 1
                else:
                    seen[(a, b)] = seen.get((a, b), 0) + 1
        assert not [v for v in seen.values() if v != 0], (
            f"{version}: start cell is not a consistently oriented closed surface")

        n = np.cross(start.vectors[:, 1] - start.vectors[:, 0],
                     start.vectors[:, 2] - start.vectors[:, 0])
        assert np.abs(n.sum(axis=0)).max() < 1e-2, f"{version}: normals do not cancel"


def test_start_ground_swaps_generation_0_and_nothing_else():
    blocks, _, plain = _parts("none")
    _, _, started = _parts("start")

    assert len(plain["meshes"]) == len(started["meshes"]), (
        "start cells replace cells, they are not extra parts")
    assert plain["cells"] == started["cells"]

    n0 = sum(1 for _, _, t in plain["cells"] if t == 0)
    n_start = sum(1 for m in started["meshes"]
                  if len(m.data) == len(blocks["start"].data))
    assert n_start == n0, f"{n_start} start cells placed, expected {n0}"

    # the model now has a flat bottom instead of meeting the bed at a point
    sp = np.concatenate([m.points.reshape(-1, 3) for m in started["meshes"]])
    low = sp[np.isclose(sp[:, 2], sp[:, 2].min(), atol=1e-6)]
    assert len(low) > 8 * n0, "no flat first layer"
    pp = np.concatenate([m.points.reshape(-1, 3) for m in plain["meshes"]])
    assert abs(sp[:, 2].min() - pp[:, 2].min()) < 1e-6, "the model changed height"


def test_plate_alone_welds_into_the_generation_0_cells():
    """With no footings the plate has to reach up into the cells itself."""
    blocks, _, model = _parts("none")
    plate, cx, cy, r = make_circular_base(model, blocks, thickness=3.0)
    top = float(plate.points.reshape(-1, 3)[:, 2].max())

    verts = blocks["cell"].vectors.reshape(-1, 3)
    ground = [(x, y) for x, y, t in model["cells"] if t == 0]
    assert ground
    for x, y in ground:
        v = verts + np.array([x * UNIT, y * UNIT, 0.0])
        inside = (v[:, 2] <= top) & (np.hypot(v[:, 0] - cx, v[:, 1] - cy) <= r)
        assert inside.any(), f"cell at {(x, y)} does not reach into the plate"


def test_a_plate_too_thin_to_reach_the_cells_is_refused():
    blocks, _, model = _parts("none")
    cell_bottom = float(blocks["cell"].points.reshape(-1, 3)[:, 2].min())
    weld = cell_bottom + 0.25

    try:
        make_circular_base(model, blocks, thickness=0.3, weld_to=weld)
    except ValueError as exc:
        assert "touch nothing" in str(exc)
    else:
        raise AssertionError("a plate that reaches nothing should be refused")

    # thick enough is accepted, and so is any thickness when footings bridge it
    make_circular_base(model, blocks, thickness=1.0, weld_to=weld)
    make_circular_base(model, blocks, thickness=0.3, weld_to=None)


def test_base_choice_survives_the_command_line():
    import tempfile
    from conway3d_stl import main as build_main
    expect = {}
    with tempfile.TemporaryDirectory() as d:
        for mode in ["cells", "none", "plate", "both"]:
            m = build_main(["--pattern", "r_pentomino", "--frames", "6",
                            "--base", mode, "--binary",
                            "--out", os.path.join(d, f"{mode}.stl")])
            expect[mode] = len(m["meshes"])
            assert os.path.getsize(os.path.join(d, f"{mode}.stl")) > 0

    # build_model's mesh list excludes the plate, so plate == none and both == cells
    assert expect["none"] == expect["plate"], "the plate is not a placed part"
    assert expect["cells"] == expect["both"], "footings should not depend on the plate"
    assert expect["cells"] > expect["none"], "footings were not placed"


# ------------------------------------------------------------ boundary modes

def _normalise(rec):
    """Live cells of each frame, moved to the origin, so grids of any size compare."""
    out = []
    for fr in rec:
        ys, xs = np.nonzero(fr)
        out.append(() if len(ys) == 0 else
                   tuple(sorted(zip(ys - ys.min(), xs - xs.min()))))
    return out


# Patterns planted with at least `frames` of clear margin on every side, so
# the worst-case growth is already there and nothing needs adding.
ROOMY = ["glider", "r_pentomino", "b_heptomino", "pi_heptomino", "acorn"]


def test_grow_leaves_a_roomy_grid_alone():
    for name in ROOMY:
        grid = getattr(init_grids, name)
        assert grow_pad(grid, DEFAULT_FRAMES) == ((0, 0), (0, 0)), (
            f"{name} already has room, growing should be a no-op")
        assert room_to_grow(grid, DEFAULT_FRAMES).shape == grid.shape

    empty = np.zeros((8, 8), dtype=int)
    assert room_to_grow(empty, 10).shape == (8, 8), "nothing alive, nothing to grow"


def test_grow_leaves_the_outermost_ring_dead_for_the_whole_run():
    """That is what makes the neighbour counts exact rather than merely roomy."""
    for size in (7, 11, 15, 21):
        grid = init_grids.from_art(init_grids.LIFE_ART["pi_heptomino"], size)
        rec = life_run(grid, DEFAULT_FRAMES, boundary="grow")
        for side in (rec[:, :1, :], rec[:, -1:, :], rec[:, :, :1], rec[:, :, -1:]):
            assert not side.any(), (
                f"a {size}x{size} start grew too little, the edge came alive")


def test_grow_from_a_cramped_grid_equals_a_huge_fixed_one():
    for name, art in init_grids.LIFE_ART.items():
        n = frames_for(name)
        cramped = life_run(init_grids.from_art(art, 11), n, boundary="grow")
        enormous = life_run(init_grids.from_art(art, 241), n, boundary="wall")
        assert _normalise(cramped) == _normalise(enormous), (
            f"{name}: growing from a cramped grid differs from a huge one")


def test_wall_clips_and_grow_does_not():
    art = init_grids.LIFE_ART["pi_heptomino"]
    full = 609                       # pi heptomino at 22 generations

    walled = plan_structure(life_run(init_grids.from_art(art, 15), DEFAULT_FRAMES,
                                     boundary="wall"))
    assert len(walled["cells"]) < full, "a 15x15 wall should cost cells"

    grown = plan_structure(life_run(init_grids.from_art(art, 15), DEFAULT_FRAMES,
                                    boundary="grow"))
    assert len(grown["cells"]) == full, "growing should recover the whole pattern"


def test_boundary_default_changes_nothing_that_ships():
    """Every shipped pattern is clear of its own boundary, so the modes agree."""
    for name, frames in ([(n, frames_for(n)) for n in init_grids.LIFE_ART]
                         + [("two_glider", 17)]):
        grid = getattr(init_grids, name)
        a = plan_structure(life_run(grid, frames, boundary="wall"))
        b = plan_structure(life_run(grid, frames, boundary="grow"))
        assert len(a["cells"]) == len(b["cells"]), f"{name}: cell count differs"
        assert len(a["rungs"]) == len(b["rungs"]), f"{name}: rung count differs"


def test_unknown_boundary_is_refused():
    try:
        life_run(init_grids.r_pentomino, 5, boundary="toroidal")
    except ValueError as exc:
        assert "toroidal" in str(exc)
    else:
        raise AssertionError("an unknown boundary should not be accepted")


# ------------------------------------------------- patterns that vanish

def test_vanishing_patterns_die_exactly_when_documented():
    """Each must still be alive the generation before, and empty on the dot."""
    for name, gen in KNOWN_VANISHING.items():
        seed = np.pad(init_grids.from_art(init_grids.LIFE_ART[name]), 30)
        rec = life_run(seed, gen + 3, boundary="wall")
        assert rec[gen - 1].sum() > 0, f"{name} was already dead before generation {gen}"
        assert rec[gen].sum() == 0, (
            f"{name} still has {int(rec[gen].sum())} cells at generation {gen}")


def test_vanishing_patterns_make_a_model_that_closes_at_the_top():
    """Run for exactly its lifetime, the top layer is the last living one."""
    for name, gen in KNOWN_VANISHING.items():
        grid = getattr(init_grids, name)
        rec = life_run(grid, gen)
        assert rec[-1].sum() > 0, f"{name}: the last frame should still be alive"
        plan = plan_structure(rec)
        assert plan["cells"], f"{name}: nothing to build"
        top = max(t for _, _, t in plan["cells"])
        assert top == gen - 1, f"{name}: top layer is {top}, expected {gen - 1}"


def test_long_period_oscillators_repeat_and_not_sooner():
    """Above period four the column repeats only every p layers."""
    for name in ["octagon2", "figure_eight", "kok_galaxy", "tumbler",
                 "pentadecathlon"]:
        period = KNOWN_BEHAVIOUR[name][0]
        assert period > 4, f"{name} is not a long period oscillator"
        seed = np.pad(init_grids.from_art(init_grids.LIFE_ART[name]), 25)
        rec = life_run(seed, period + 1, boundary="wall")
        assert np.array_equal(rec[period], rec[0]), f"{name} did not return at {period}"
        for shorter in range(1, period):
            assert not np.array_equal(rec[shorter], rec[0]), (
                f"{name} repeats at {shorter}, so its period is not {period}")


def _canonical(sub):
    """Smallest of the eight rotations and reflections of a cropped pattern."""
    best = None
    for k in range(4):
        r = np.rot90(sub, k)
        for m in (r, r[:, ::-1]):
            key = (m.shape, m.astype(np.int8).tobytes())
            if best is None or key < best:
                best = key
    return best


def test_the_named_patterns_are_all_distinct_objects():
    """No two names may be one object caught at different times or angles.

    A pattern one generation into another's run is not a new pattern, and
    neither is the same thing rotated, so both are compared away.
    """
    seen = {}
    for name in init_grids.LIFE_ART:
        rec = life_run(getattr(init_grids, name), 18, boundary="wall")
        phases = set()
        for fr in rec:
            ys, xs = np.nonzero(fr)
            if len(ys) == 0:
                continue
            phases.add(_canonical(fr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
        for other, theirs in seen.items():
            assert not (phases & theirs), f"{name} and {other} are the same object"
        seen[name] = phases


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
