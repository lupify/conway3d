# conway3d

Turns a run of Conway's Game of Life into a single STL that 3D prints in one
piece, with no assembly and no support material.

The Game of Life is two dimensional and unfolds over time. This treats time as
the third dimension: generation 0 sits on the build plate and each later
generation is stacked 10 mm above the one before, so the finished print is the
whole history of the pattern, readable from bottom to top.

<!-- Add a photo or a render here. `preview.py` produces one. -->

## What the model is made of

Every live cell becomes one solid part placed on a 10 mm cubic lattice at
`(x, y, generation)`. Nothing else about the simulation is represented.

Generations do not touch each other directly. A cell part is 7.15 mm tall on a
10 mm pitch, so there is a 2.85 mm gap between one generation and the next.
That gap is bridged by **rungs**: wherever a live cell has a live *neighbour*
in the next generation, a strut is placed linking the two. There are two rung
parts, one for an orthogonal step and a longer one for a diagonal step.

What sits underneath is up to you: a square **footing** part beneath each cell
of generation 0, a round **plate** under the whole thing, both, or nothing at
all (see below).

### Rungs never go straight up

A rung is only ever placed between a cell and one of the eight positions
*around* it in the next generation. A cell is never joined to the same position
directly above itself, even when it survives in place.

This is deliberate, and it is the point of the object. Under the Life rules no
part of the structure ever rests on a single purely vertical support, so the
print reads as though it should not be able to stand up.

It still holds together, and that is a property of the rules rather than luck.
A cell alive at generation `t+1` had at least two live neighbours at `t`: three
to be born, two or three to survive. Those neighbours are by definition in the
eight surrounding positions, so at least one rung always reaches down to the
previous generation. By induction every cell connects to generation 0 and so to
the plate.

What this does *not* guarantee is that the model is a single object. Two seed
clusters that never interact grow two separate towers. `check_connectivity()`
reports this, the test suite checks it, and a plate joins them anyway if you
use one.

### What goes underneath

`--base` takes one of four values.

| `--base` | what you get |
|---|---|
| `cells` | a square footing under each generation 0 cell. The default, and what the original project did |
| `none` | nothing. Cells and rungs only, for a piece to turn over in your hand |
| `plate` | a round plate and no footings |
| `both` | footings and plate |

Patterns that spread as they grow are top heavy, which is what the plate is
for. It is sized by default to cover the *whole* model in plan view rather than
just the cells touching the ground, so the centre of mass sits inside the
footprint however far the shape leans. It starts at the height of the underside
of a footing, so it fuses with the footings, and with `plate` it reaches up
into the generation 0 cells directly.

That last case has a floor. Generation 0 cells hang down to z = −3.575 and the
plate starts at z = −4, so a plate thinner than 0.425 mm would sit entirely
below them and touch nothing. Anything under 0.67 mm is refused with a message
saying so. It is not a problem with `both`, where the footings bridge the gap.

With `none` the lowest point of the model is the underside of a generation 0
cell, at z = −3.575, and nothing is holding it to the build plate. It is a
shape to hold, not one to stand up. Note also that only the plate can join
separate towers into one object: a pattern that grows from two seeds which
never interact is genuinely two pieces without it, and the tools will tell you
so.

## Sizes

Height depends only on the number of generations:

```
height in mm = (generations - 1) x 10 + 7.575
```

so 17 generations is 167.6 mm, 20 is 197.6 mm and 22 is 217.6 mm. Width depends
on how far the pattern spreads.

Part dimensions, `model_stls/version3`, in millimetres:

| part | footprint | height range | triangles |
|---|---|---|---|
| cell | 7.20 across | −3.575 to +3.575 | 812 |
| base | 10.4 square | −4.000 to 0.000 | 900 |
| rung, orthogonal | reaches 0.69 to 9.93 out, 2.50 wide | +2.379 to +7.285 | 622 |
| rung, diagonal | reaches 1.30 to 14.34 out, 2.50 wide | +2.055 to +7.269 | 748 |

The build plate is at z = −4.

## Installing

```
pip install -r requirements.txt
```

Needs numpy, scipy and numpy-stl. Pillow is only used for `--image`, and
matplotlib only for previews and plots. The drawing tool uses tkinter, which
ships with Python but is a separate package on some Linux distributions
(`python3-tk`).

## Building a model

```
python conway3d_stl.py --pattern tree_fork --frames 22 \
    --base both --center --binary --out output_stls/tree_fork.stl

# the same tree with nothing underneath, to hold rather than stand up
python conway3d_stl.py --pattern tree_fork --frames 22 \
    --base none --binary --out output_stls/tree_fork_handheld.stl
```

The initial condition comes from exactly one of:

- `--pattern NAME` — a grid defined in `init_grids.py`
- `--array FILE` — a pattern file of on/off cells, see below
- `--image FILE` — a 1-bit bitmap, where black pixels are live cells

Other options:

| flag | effect |
|---|---|
| `--frames N` | generations to run, counting the initial one. Sets the height. |
| `--base MODE` | `cells`, `none`, `plate` or `both`. Default `cells` |
| `--base-radius`, `--base-thickness`, `--base-segments` | override the plate, default radius covers the model, 3 mm thick, 180 facets |
| `--center` | move the model over the origin, ready to slice |
| `--binary` | write a binary STL. Roughly four times smaller than the default ASCII |
| `--model-stls DIR` | choose the part set, default `./model_stls/version3` |
| `--unit MM` | lattice pitch, must match the parts |
| `--pad N` | surround the starting grid with N dead cells |
| `--plot`, `--verbose` | diagnostic plots, and a line per rung placed |

It prints the cell and rung counts, what base was used, the triangle count and
how many separate pieces the result is in.

## Drawing your own

```
python designer.py
python designer.py --load my_pattern.txt
python designer.py --rows 61 --cols 61 --frames 22
```

A small window. Set the size of the domain and press **Resize domain**, then
paint live cells with the left mouse button and erase with the right. Dragging
paints continuously.

The **base** dropdown offers the same four choices as `--base` on the command
line, and the drawing updates to match.

The side panel keeps a running summary as you draw: how many cell and rung
parts the model will need, how tall and wide it will print, the footing count
and plate diameter where they apply, and how many separate pieces it will be
in. It warns you when

- the pattern reaches the edge of the domain, which silently clips it and means
  you are no longer simulating Life, so the domain needs to be bigger;
- nothing survives to the last generation;
- the model would come out in several pieces, which only a plate can join;
- the base is set to `none`, so nothing holds the result on the build plate.

**Preview 3D** draws the structure without building geometry. **Generate STL…**
writes the file, on a background thread so the window stays responsive.
**Open…** and **Save as…** read and write the pattern files described next, so
a design can be put away and picked up again. Ctrl+O and Ctrl+S work too.

## Pattern files

A pattern file holds nothing but the on/off cells. `init_grids.load_pattern()`
reads two forms.

**Text**, one line per row:

```
! lines starting with an exclamation mark are comments
.#.
..#
###
```

`#`, `O`, `o`, `*`, `X`, `x` and `1` are live. `.`, `0`, `-`, `_` and space are
dead. Values may be separated by commas, tabs or spaces, so a 0/1 grid exported
from a spreadsheet loads unchanged; otherwise each character is one cell. Short
rows are padded with dead cells.

Every line is a row, blank ones included, so the file's shape *is* the domain
and saving then loading returns exactly what you drew. This also reads the
plaintext `.cells` files used elsewhere in the Life world.

**NumPy**, a `.npy` file or a `.npz` whose first array is used. Any non-zero
value counts as live.

`init_grids.save_pattern(grid, path)` writes text, or NumPy if the name ends in
`.npy`.

## Previewing

```
python preview.py --pattern tree_crown --frames 22 --base-radius auto --out t.png
python preview.py --array my_pattern.txt --frames 22 --out mine.png
python preview.py --seeds candidates.npz --frames 22 --cols 4 --out grid.png
```

Draws a dot per cell and a line per rung, which is much faster than building the
STL and enough to judge a shape. It reports the true printed dimensions.

## The pattern library

`init_grids.py` holds named grids usable with `--pattern`.

`two_glider` is two gliders that collide at generation 10 and settle into a pond
still life at generation 15. It is the array form of
`initial_sates/twoGlider_pond.png`.

Five seeds are shaped to print as trees, written as ASCII art in `TREE_ART` and
sized for 22 generations. A cluster that persists forms the trunk; gliders
leaving it lean outward at a fixed angle as they climb, and that is what draws
the branches.

| pattern | cell parts | rungs | plate | triangles | binary STL |
|---|---|---|---|---|---|
| `tree_fork` | 210 | 539 | 91 mm | 547,358 | 27 MB |
| `tree_cross` | 257 | 672 | 91 mm | 673,720 | 34 MB |
| `tree_slender` | 239 | 609 | 110 mm | 619,360 | 31 MB |
| `tree_crown` | 252 | 628 | 110 mm | 643,912 | 32 MB |
| `tree_pine` | 609 | 1,636 | 194 mm | 1,615,888 | 81 MB |

Triangle and file sizes are for 22 generations with `--base both`; the plate
adds 720 triangles of its own. All five are 217.6 mm tall. The first four are slender and
open. `tree_pine` grows from a pi heptomino and is dense all the way up, a
conifer rather than a bare tree, and by far the biggest print.

Also present, from earlier work: `glider`, `bar`, `k4grid`, `k4grid2`,
`k4grid_rev`, `KD` and `nineoscillator`.

## Searching for shapes

`tree_search.py` sweeps initial conditions looking for a given silhouette. It
scores a space-time history on how the outline widens with height, whether the
trunk splits into separate clusters, and how slender those branches stay.

```
python tree_search.py --frames 22 --top 16 --save candidates.npz
python preview.py --seeds candidates.npz --frames 22 --out candidates.png
```

Seeds come from known methuselahs, from exhaustive sweeps of mirror, four-fold
and eight-fold symmetric boxes, and from random asymmetric boxes.

Two things are worth knowing before you use it. Slenderness has to be a hard
rejection rather than a weighting, because a chaotically growing blob scores
well on every "widens upward" measure while printing as a solid cone. And at
around 22 generations Life really only produces four silhouettes: a leaning
column, a Y-fork, a dense cone and a goblet. A tree with boughs at several
heights does not appear at that height.

## Part sets

`model_stls/` holds three generations of the four parts. Each directory must
contain `base.stl`, `cell.stl`, `rung_adj.stl` and `rung_diag.stl`.

| set | base | cell | rung, orthogonal | rung, diagonal |
|---|---|---|---|---|
| `version1` | 822 | 812 | 900 | 900 |
| `version2` | 924 | 812 | 774 | 732 |
| `version3` | 900 | 812 | 622 | 748 |

`version3` is the default. All three are exercised by the tests. They differ in
detail and triangle count, not in lattice pitch, so any of them can be swapped
in with `--model-stls`.

## Printing notes

- **Height.** 22 generations is 217.6 mm, which is over the 210 mm limit of
  some printers. Drop to 20 generations for 197.6 mm.
- **Overhangs.** The rungs are steep. The orthogonal one rises 27.1° above
  horizontal and the diagonal one 21.4°. They are short, 2.5 mm wide and land
  on a cell at each end, but there are a great many of them.
- **Print in place.** There is nothing to assemble and no support is intended.
  The model is one connected solid provided the tools say it is one piece.
- **`--base none` is for handling, not for printing straight off.** With
  nothing underneath, generation 0 cells meet the bed at a rounded point.
  Print it with footings and remove them, or use supports.
- **The STL is a soup of overlapping closed solids**, not a single manifold
  surface. Slicers union it without complaint, but mesh validators will report
  it as not closed. That is expected.
- **File size.** Use `--binary`. ASCII STL is around four times larger for the
  same geometry, and these models run to hundreds of thousands of triangles.
- Generated models go to `output_stls/`, which is deliberately not tracked by
  git.

## Is it really Conway's Life?

Yes, and the test suite checks it rather than asserting it.

- The transition function is tested on all 18 combinations of cell state and
  neighbour count, which is B3/S23 by exhaustion, and the neighbourhood is
  confirmed to be the eight surrounding cells with the centre excluded.
- Five still lifes, four oscillators and two spaceships must hold their known
  periods, and the spaceships their exact displacement per period. Each is also
  checked not to repeat sooner.
- The R-pentomino is run for 1103 generations on a field 641 cells wide and must
  settle at 116 cells, the documented figure. The field is wide enough that the
  six escaping gliders never reach an edge, so that run is infinite-plane Life.
- Each shipped pattern is evolved again on a grid nearly three times wider and
  must produce an identical history, proving it is not being clipped by its
  boundary.
- The placed cells are read back out of the model and compared against the
  simulation, so the print is a faithful record with one cell part per live cell.

The simulation uses a finite grid with dead cells outside it, which equals
infinite Life only while the pattern stays clear of the edge. That is why the
clipping check matters, and why the drawing tool warns about it.

## Tests

```
python test_pipeline.py      # or: python -m pytest test_pipeline.py -q
```

37 tests covering the rules, the parts, model assembly, the base options,
pattern files and the shipped patterns. Beyond the Life checks above they confirm that
every building block is a closed surface with no non-manifold edges, that every
placed rung physically reaches both cells it links and aims within 3° of its
target, that models come out as one grounded piece, and that written STLs
round trip, that each base mode places exactly the parts it should, and that a
plate too thin to reach the cells is refused. One test pins the placement logic to the geometry of an STL
produced before this repository existed.

## Layout

```
conway3d_stl.py     the pipeline: run Life, place parts, write the STL
designer.py         draw a pattern with the mouse and build it
preview.py          render a model to PNG without building geometry
tree_search.py      search initial conditions for a given silhouette
init_grids.py       named patterns, the tree seeds, pattern file reading
test_pipeline.py    the test suite
model_stls/         the four building blocks, three versions
initial_sates/      example bitmaps, and a 16x16 Photoshop template
output_stls/        generated models land here, not tracked
```
