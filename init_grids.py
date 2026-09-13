import re

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
# initial_states/twoGlider_pond.png.
two_glider = np.zeros((32, 32), dtype=int)
for _r, _c in [(8, 12),
               (9, 11),
               (10, 11), (10, 12), (10, 13),
               (17, 13),
               (18, 12), (18, 13),
               (19, 12), (19, 14)]:
    two_glider[_r, _c] = 1
del _r, _c


# ------------------------------------------------------- common Life patterns
#
# The standard menagerie, each one a well known object with a documented
# period and, for the spaceships, a documented displacement.  What each looks
# like as a print follows from what it does in the plane, because time is the
# vertical axis:
#
#   still life   an unchanging shape, so a straight prism of that cross section
#   oscillator   the same shape returning every p generations, so a column
#                that repeats with a vertical period of p * unit
#   spaceship    a shape translating steadily, so a leaning column whose angle
#                is set by how far it moves per generation
#   methuselah   a small seed that stays chaotic for a long time, so an
#                irregular, branching mass
#   gun          a still machine that emits spaceships, so a fixed core with
#                leaning columns streaming away from it at a regular interval
#
# The grid each one is planted in has to be big enough that the pattern never
# reaches its edge, or it is being clipped and is no longer Life.  Travelling
# and growing patterns therefore get more room than static ones.

LIFE_ART = {
    # ---- still lifes: unchanging, so a plain vertical prism
    "block":        ["##",
                     "##"],
    "beehive":      [".##.",
                     "#..#",
                     ".##."],
    "loaf":         [".##.",
                     "#..#",
                     ".#.#",
                     "..#."],
    "boat":         ["##.",
                     "#.#",
                     ".#."],
    "tub":          [".#.",
                     "#.#",
                     ".#."],

    # ---- oscillators: a column repeating with the period of the object
    "blinker":      ["###"],
    "toad":         [".###",
                     "###."],
    "beacon":       ["##..",
                     "##..",
                     "..##",
                     "..##"],
    "pulsar":       ["..###...###..",
                     ".............",
                     "#....#.#....#",
                     "#....#.#....#",
                     "#....#.#....#",
                     "..###...###..",
                     ".............",
                     "..###...###..",
                     "#....#.#....#",
                     "#....#.#....#",
                     "#....#.#....#",
                     ".............",
                     "..###...###.."],
    "pentadecathlon": ["..#....#..",
                       "##.####.##",
                       "..#....#.."],

    # ---- spaceships: a leaning column
    "glider":       [".#.",
                     "..#",
                     "###"],
    "lwss":         [".####",
                     "#...#",
                     "....#",
                     "#..#."],
    "mwss":         ["...#..",
                     ".#...#",
                     "#.....",
                     "#....#",
                     "#####."],
    "hwss":         ["...##..",
                     ".#....#",
                     "#......",
                     "#.....#",
                     "######."],

    # ---- methuselahs: small seeds that stay chaotic for a long time
    "r_pentomino":  [".##",
                     "##.",
                     ".#."],
    "acorn":        [".#.....",
                     "...#...",
                     "##..###"],
    "diehard":      ["......#.",
                     "##......",
                     ".#...###"],
    "b_heptomino":  ["###.",
                     ".###",
                     "..#."],
    "pi_heptomino": ["###",
                     "#.#",
                     "#.#"],

    # ---- a gun: a fixed core shedding gliders at a regular interval
    "gosper_glider_gun": [
        "........................#...........",
        "......................#.#...........",
        "............##......##............##",
        "...........#...#....##............##",
        "##........#.....#...##..............",
        "##........#...#.##....#.#...........",
        "..........#.....#.......#...........",
        "...........#...#....................",
        "............##......................"],
}

# How much room each one needs to run without touching the edge of its grid.
# A pattern spreads by at most one cell per generation, so a static object
# needs almost nothing and a gun needs room for everything it has emitted.
LIFE_SIZE = {
    "pulsar": 33, "pentadecathlon": 33,
    "glider": 49, "lwss": 61, "mwss": 61, "hwss": 61,
    "r_pentomino": 61, "acorn": 61, "diehard": 49,
    "b_heptomino": 61, "pi_heptomino": 61,
    "gosper_glider_gun": 91,
}
DEFAULT_GRID_SIZE = 25


def from_art(rows, size=DEFAULT_GRID_SIZE):
    """Turn rows of '#' and '.' into a centred grid of the given size."""
    art = np.array([[1 if c == "#" else 0 for c in row] for row in rows], dtype=int)
    if size < max(art.shape):
        size = max(art.shape)
    grid = np.zeros((size, size), dtype=int)
    r0 = size // 2 - art.shape[0] // 2
    c0 = size // 2 - art.shape[1] // 2
    grid[r0:r0 + art.shape[0], c0:c0 + art.shape[1]] = art
    return grid


for _name, _art in LIFE_ART.items():
    globals()[_name] = from_art(_art, LIFE_SIZE.get(_name, DEFAULT_GRID_SIZE))
del _name, _art


# ------------------------------------------------------------- pattern files
#
# A pattern file holds nothing but the on/off cells.  Two forms are read:
#
#   text   one line per row.  '#', 'O', 'o', '*', 'X', 'x' and '1' are live;
#          '.', '0', '-', '_' and space are dead.  Lines beginning with '!'
#          are comments.  Values may be separated by commas, tabs or spaces,
#          otherwise each character is one cell.  Short rows are padded with
#          dead cells, so ragged files load fine.  Every line is a row,
#          blank ones included, so the file's shape is the domain and a save
#          then load returns exactly what you drew.  This reads the plaintext
#          ".cells" files used elsewhere in the Life world.
#
#   numpy  .npy, or .npz where the first array in the file is used.
#
# Anything non-zero counts as live, so a 0/1 CSV exported from a spreadsheet
# loads without conversion.

LIVE_CHARS = set("#Oo*Xx1")
DEAD_CHARS = set(".0-_ ")


def parse_art(text):
    """Parse the text pattern format into a 2-D array of 0s and 1s."""
    rows = []
    for line in text.splitlines():
        if line.lstrip().startswith("!"):
            continue
        stripped = line.strip()
        tokens = [t for t in re.split(r"[,\t ]+", stripped) if t]
        if len(tokens) > 1 and all(len(t) == 1 for t in tokens):
            cells = tokens                 # "0, 1, 0" or "0 1 0"
        else:
            cells = list(line.rstrip("\n"))  # one character per cell
        row = []
        for ch in cells:
            if ch in LIVE_CHARS:
                row.append(1)
            elif ch in DEAD_CHARS or ch == "":
                row.append(0)
            else:
                raise ValueError(f"unrecognised cell character {ch!r}")
        rows.append(row)

    if not rows:
        raise ValueError("pattern file contains no cells")

    width = max(len(r) for r in rows)
    return np.array([r + [0] * (width - len(r)) for r in rows], dtype=int)


def load_pattern(path):
    """Read a pattern file, text or numpy, into a 2-D array of 0s and 1s."""
    path = str(path)
    if path.endswith(".npy"):
        grid = np.load(path)
    elif path.endswith(".npz"):
        with np.load(path) as d:
            grid = d[d.files[0]]
    else:
        with open(path) as fh:
            return parse_art(fh.read())

    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"expected a 2-D grid, got shape {grid.shape}")
    return (grid != 0).astype(int)


def save_pattern(grid, path, comment=None):
    """Write a pattern out.  Text unless the name ends in .npy."""
    grid = (np.asarray(grid) != 0).astype(int)
    if str(path).endswith(".npy"):
        np.save(path, grid)
        return
    lines = []
    if comment:
        lines += [f"! {line}" for line in str(comment).splitlines()]
    lines += ["".join("#" if v else "." for v in row) for row in grid]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
