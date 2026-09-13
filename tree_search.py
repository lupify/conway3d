"""Search for initial conditions whose space-time history looks like a tree.

Time is the print's vertical axis, so a tree needs a narrow, compact start that
widens and splits into separate clusters as the generations go up.  Gliders and
spaceships are what make the branches: they lean away from the trunk at a fixed
angle while they climb.

    python tree_search.py --frames 22 --top 12
"""

import argparse

import numpy as np
from scipy import ndimage


def step(g):
    """One generation, zero boundary, matching conway3d_stl.life_run."""
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


# ------------------------------------------------------------------ candidates

METHUSELAHS = {
    "r_pentomino":  [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "b_heptomino":  [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1), (0, 3)],
    "pi_heptomino": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)],
    "acorn":        [(0, 1), (1, 3), (2, 0), (2, 1), (2, 4), (2, 5), (2, 6)],
    "diehard":      [(0, 6), (1, 0), (1, 1), (2, 1), (2, 5), (2, 6), (2, 7)],
    "thunderbird":  [(0, 0), (0, 1), (0, 2), (2, 1), (3, 1), (4, 1)],
    "herschel":     [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (2, 0)],
    "glider":       [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "lwss":         [(0, 0), (0, 3), (1, 4), (2, 0), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
}


def named_seeds(size):
    out = {}
    for name, cells in METHUSELAHS.items():
        g = np.zeros((size, size), dtype=np.int8)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0 = size // 2 - (max(rs) + min(rs)) // 2
        c0 = size // 2 - (max(cs) + min(cs)) // 2
        for r, c in cells:
            g[r0 + r, c0 + c] = 1
        out[name] = g
    return out


def _place(seeds, win, size, box):
    r0 = c0 = size // 2 - box // 2
    seeds[:, r0:r0 + box, c0:c0 + box] = win
    return seeds


def quad_seeds(size, box=7):
    """Seeds mirrored about both axes, so the tree grows four branches."""
    half = (box + 1) // 2
    nfree = half * half
    total = 1 << nfree
    bits = ((np.arange(total)[:, None] >> np.arange(nfree)) & 1).astype(np.int8)
    q = bits.reshape(total, half, half)
    top = np.concatenate([q, q[:, :, ::-1][:, :, box % 2:]], axis=2)
    win = np.concatenate([top, top[:, ::-1, :][:, box % 2:, :]], axis=1)
    return _place(np.zeros((total, size, size), dtype=np.int8), win, size, box)


def octo_seeds(size, box=9):
    """Seeds with full eight-fold symmetry: four branches plus diagonals."""
    half = (box + 1) // 2
    iu = np.triu_indices(half)
    nfree = len(iu[0])
    total = 1 << nfree
    bits = ((np.arange(total)[:, None] >> np.arange(nfree)) & 1).astype(np.int8)
    q = np.zeros((total, half, half), dtype=np.int8)
    q[:, iu[0], iu[1]] = bits
    q = np.maximum(q, np.transpose(q, (0, 2, 1)))
    top = np.concatenate([q, q[:, :, ::-1][:, :, box % 2:]], axis=2)
    win = np.concatenate([top, top[:, ::-1, :][:, box % 2:, :]], axis=1)
    return _place(np.zeros((total, size, size), dtype=np.int8), win, size, box)


def random_seeds(size, box, n, density=0.4, seed=0):
    """Random asymmetric seeds.  Real trees are not symmetric either."""
    rng = np.random.default_rng(seed)
    win = (rng.random((n, box, box)) < density).astype(np.int8)
    return _place(np.zeros((n, size, size), dtype=np.int8), win, size, box)


def mirror_seeds(size, box=5):
    """Every seed in a box x box window with left-right mirror symmetry.

    Life preserves symmetry, so these stay symmetric for ever, which is what
    makes them read as a tree rather than as debris.
    """
    half = (box + 1) // 2
    nfree = box * half
    total = 1 << nfree
    seeds = np.zeros((total, size, size), dtype=np.int8)
    r0 = size // 2 - box // 2
    c0 = size // 2 - box // 2
    bits = ((np.arange(total)[:, None] >> np.arange(nfree)) & 1).astype(np.int8)
    tile = bits.reshape(total, box, half)
    win = np.concatenate([tile, tile[:, :, ::-1][:, :, box % 2:]], axis=2)
    seeds[:, r0:r0 + box, c0:c0 + box] = win
    return seeds


# --------------------------------------------------------------------- scoring

def cheap_metrics(hist, margin=2):
    """Per-seed population, silhouette radius profile and boundary safety."""
    b, f, h, w = hist.shape
    pop = hist.sum(axis=(2, 3)).astype(np.float64)

    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = (hist * dist).max(axis=(2, 3))  # furthest live cell each frame

    edge = np.zeros_like(hist, dtype=bool)
    edge[:, :, :margin, :] = edge[:, :, -margin:, :] = True
    edge[:, :, :, :margin] = edge[:, :, :, -margin:] = True
    safe = ~(hist.astype(bool) & edge).any(axis=(1, 2, 3))
    return pop, radius, safe


def cluster_counts(hist_one):
    """8-connected clusters per frame: the trunk splitting into branches."""
    s = ndimage.generate_binary_structure(2, 2)
    return np.array([ndimage.label(fr, structure=s)[1] for fr in hist_one])


def tree_score(pop, radius, clusters, spread_last=0.0,
               max_density=4.5, max_final_pop=40):
    """Higher is more tree shaped.

    What reads as a tree is open space between branches, not bulk.  A dense
    growing blob scores well on every "widens upward" measure while looking
    like a cone, so slenderness is enforced rather than merely rewarded.
    """
    f = len(pop)
    lo, hi = slice(0, f // 3), slice(2 * f // 3, f)

    if pop[-1] < 8 or radius[-1] <= radius[0]:
        return -1e9, {}

    bands = np.array_split(clusters, 4)
    means = np.array([b.mean() for b in bands])
    if means[0] > 1.6:                      # must leave the ground as one trunk
        return -1e9, {}

    top_branches = float(clusters[hi].mean())
    tips = int(clusters[-1])
    density = float(pop[hi].mean() / max(top_branches, 1))
    if tips < 3 or top_branches < 2.5:      # a fork is the minimum
        return -1e9, {}
    if density > max_density:               # otherwise it is a cone, not a tree
        return -1e9, {}
    if pop[-1] > max_final_pop:             # a heavy canopy closes the gaps
        return -1e9, {}

    grow = float(np.corrcoef(np.arange(f), radius)[0, 1])
    flare = float((radius[hi].mean() + 1) / (radius[lo].mean() + 1))
    stepped = float(np.clip(np.diff(means), 0, None).sum())
    trunk = float(1.0 / (1.0 + radius[lo].mean()))

    score = (2.0 * np.log(tips) + 1.6 * np.log(1 + spread_last)
             + 2.0 * grow + 1.5 * np.log(flare) + 1.2 * stepped
             - 1.0 * np.log(density) + 1.5 * trunk
             - 0.6 * np.log(max(pop.sum(), 1)))
    return score, {"grow": grow, "flare": flare, "branch": top_branches,
                   "density": density, "stepped": stepped, "tips": tips,
                   "spread": spread_last, "canopy": float(pop[hi].mean()),
                   "cells": int(pop.sum()), "final_pop": int(pop[-1]),
                   "max_radius": float(radius.max())}


def tip_spread(frame):
    """Mean distance of the final clusters from the trunk axis, in cells."""
    s = ndimage.generate_binary_structure(2, 2)
    lab, n = ndimage.label(frame, structure=s)
    if n == 0:
        return 0.0
    cen = np.array(ndimage.center_of_mass(frame, lab, range(1, n + 1)))
    mid = np.array(frame.shape) / 2
    return float(np.linalg.norm(cen - mid, axis=1).mean())


def search(frames=22, size=49, box=5, top=12, max_cells=460, min_cells=110,
           max_radius=9.5, chunk=2048, nrandom=60000):
    results = []

    pool = {}
    pool.update(named_seeds(size))
    named = set(pool)
    pools = [("mirror", mirror_seeds(size, box)),
             ("quad", quad_seeds(size, 7)),
             ("octo", octo_seeds(size, 9))]
    for i, (bx, dens) in enumerate([(5, .45), (6, .4), (6, .5), (7, .35), (7, .45)]):
        pools.append((f"rand{bx}{int(dens*100)}",
                      random_seeds(size, bx, nrandom, dens, seed=i)))

    # named patterns first
    if pool:
        keys = list(pool)
        hist = run_batch(np.array([pool[k] for k in keys]), frames)
        pop, rad, safe = cheap_metrics(hist)
        for i, k in enumerate(keys):
            if not safe[i]:
                continue
            sc, m = tree_score(pop[i], rad[i], cluster_counts(hist[i]),
                               tip_spread(hist[i][-1]))
            if m:
                results.append((sc, k, pool[k], m, hist[i]))

    # then the symmetric sweep, filtered cheaply before the costly cluster pass
    for pool_name, mirrors in pools:
      for s in range(0, len(mirrors), chunk):
        batch = mirrors[s:s + chunk]
        hist = run_batch(batch, frames)
        pop, rad, safe = cheap_metrics(hist)
        keep = (safe & (pop.sum(axis=1) >= min_cells) & (pop.sum(axis=1) <= max_cells)
                & (rad[:, -1] > rad[:, 0]) & (rad.max(axis=1) <= max_radius)
                & (pop[:, -1] >= 10))
        for i in np.nonzero(keep)[0]:
            cl = cluster_counts(hist[i])
            sc, m = tree_score(pop[i], rad[i], cl, tip_spread(hist[i][-1]))
            if m:
                results.append((sc, f"{pool_name}_{s + i}", batch[i].copy(), m,
                                (tuple(pop[i].astype(int)), tuple(cl))))

    results.sort(key=lambda r: -r[0])

    # collapse near-identical silhouettes: same population and branching profile
    seen, best = set(), []
    for sc, name, seed, m, extra in results:
        if isinstance(extra, tuple):
            sig = extra
        else:
            sig = (tuple(extra.sum(axis=(1, 2)).astype(int)),
                   tuple(cluster_counts(extra)))
        if sig in seen:
            continue
        seen.add(sig)
        best.append((sc, name, seed, m))
        if len(best) >= top:
            break
    return best, named


def profile(hist_one):
    """Compact text silhouette: width of each frame from bottom to top."""
    rows = []
    for t, fr in enumerate(hist_one):
        ys, xs = np.nonzero(fr)
        if len(xs) == 0:
            rows.append((t, 0, 0, 0))
            continue
        s = ndimage.generate_binary_structure(2, 2)
        rows.append((t, int(fr.sum()), int(xs.max() - xs.min() + 1),
                     ndimage.label(fr, structure=s)[1]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=22)
    p.add_argument("--size", type=int, default=49)
    p.add_argument("--nrandom", type=int, default=60000)
    p.add_argument("--box", type=int, default=5)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--max-cells", type=int, default=300)
    p.add_argument("--min-cells", type=int, default=110)
    p.add_argument("--max-radius", type=float, default=9.5)
    p.add_argument("--save", help="write the winning seeds to this .npz")
    args = p.parse_args()

    best, named = search(frames=args.frames, size=args.size, box=args.box,
                         top=args.top, max_cells=args.max_cells,
                         min_cells=args.min_cells, max_radius=args.max_radius,
                         nrandom=args.nrandom)

    print(f"{'rank':>4} {'name':>14} {'score':>7} {'cells':>6} {'final':>6} "
          f"{'flare':>6} {'canopy':>7} {'branch':>7} {'dens':>6} {'step':>5} {'tips':>5} {'spread':>7} {'radius':>7}")
    for i, (sc, name, seed, m) in enumerate(best, 1):
        print(f"{i:>4} {name:>14} {sc:7.2f} {m['cells']:6d} {m['final_pop']:6d} "
              f"{m['flare']:6.2f} {m['canopy']:7.2f} {m['branch']:7.2f} "
              f"{m['density']:6.1f} {m['stepped']:5.2f} {m['tips']:5d} "
              f"{m['spread']:7.1f} {m['max_radius']:7.1f}")

    if args.save:
        np.savez_compressed(args.save,
                            **{f"{i:02d}_{name}": seed
                               for i, (_, name, seed, _) in enumerate(best, 1)})
        print(f"\nwrote {args.save}")
    return best


if __name__ == "__main__":
    main()
