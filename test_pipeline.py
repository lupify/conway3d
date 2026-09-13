"""End-to-end check of the Conway -> print-in-place pipeline.

Uses two gliders colliding into a pond as the test case.  Run directly
(``python test_pipeline.py``) or under pytest.
"""

import os
import tempfile

import numpy as np
from stl import mesh

import init_grids
from conway3d_stl import (build_model, check_connectivity, life_run,
                          load_building_blocks, load_grid_from_image, save_stl)

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


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}\n      {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}\n      {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
