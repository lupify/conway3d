import numpy as np

k4grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                   [0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
                   [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                   [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                   [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) 

glider = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) 
                   
bar = np.array([[0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0]]) 
                   
k4grid2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]) 

k4grid_rev = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 1, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 1, 0],
                       [0, 0, 0, 0, 1, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]]) 
         
KD = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
               [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) 
               
                    
nineoscillator = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                           [0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
                           [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) 

# Two gliders on a collision course.  They meet around step 10 and settle into
# a "pond" still life by step 15.  This is the array form of
# initial_sates/twoGlider_pond.png.
two_glider = np.zeros((32, 32), dtype=int)
for _r, _c in [(8, 12),
               (9, 11),
               (10, 11), (10, 12), (10, 13),
               (17, 13),
               (18, 12), (18, 13),
               (19, 12), (19, 14)]:
    two_glider[_r, _c] = 1
del _r, _c


# ---------------------------------------------------------------------- trees
#
# Seeds whose space-time history looks like a tree when printed, since time is
# the vertical axis.  A cluster that persists makes the trunk; gliders and
# spaceships leaving it lean outwards as they climb, which is what draws the
# branches.  Found with tree_search.py over symmetric and random seeds.
#
# All are sized for 22 generations, which prints about 218mm tall on the
# 10mm lattice.

TREE_ART = {
    # a single trunk that splits cleanly in two.  The smallest seed of the set
    "tree_fork":     ["#####",
                      "##.##"],

    # a plain plus sign, which opens into the widest crown of the slim set
    "tree_cross":    [".#.",
                      "###",
                      ".#."],

    # a long bare trunk with a late, narrow crown
    "tree_slender":  ["...#.",
                      "..#.#",
                      "#.#..",
                      "##.#.",
                      "....#"],

    # a wider, more open crown on a short trunk
    "tree_crown":    [".###.",
                      "#...#",
                      ".#.#.",
                      "##.##"],

    # the pi heptomino: dense all the way up, a conifer rather than a bare tree
    "tree_pine":     ["###",
                      "#.#",
                      "#.#"],
}

TREE_GRID_SIZE = 49  # room for 22 generations without touching the boundary


def from_art(rows, size=TREE_GRID_SIZE):
    """Turn rows of '#' and '.' into a centred grid of the given size."""
    art = np.array([[1 if c == "#" else 0 for c in row] for row in rows], dtype=int)
    grid = np.zeros((size, size), dtype=int)
    r0 = size // 2 - art.shape[0] // 2
    c0 = size // 2 - art.shape[1] // 2
    grid[r0:r0 + art.shape[0], c0:c0 + art.shape[1]] = art
    return grid


for _name, _art in TREE_ART.items():
    globals()[_name] = from_art(_art)
del _name, _art
