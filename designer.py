"""Draw an initial condition with the mouse, then build the model from it.

A small tkinter window: pick the size of the domain, paint live cells with the
left button and erase with the right, then preview the space-time structure or
write the STL.  Patterns load from and save to the plain on/off files that
init_grids.load_pattern reads, so a design can be kept and re-opened.

    python designer.py
    python designer.py --load my_pattern.txt
    python designer.py --rows 49 --cols 49 --frames 22
"""

import argparse
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import init_grids
from conway3d_stl import (build_model, check_connectivity, life_run,
                          load_building_blocks, make_circular_base,
                          plan_structure, save_stl)

MAX_CANVAS = 640          # pixels on the long side of the drawing area
MAX_CELL_PX = 26
MIN_CELL_PX = 3
AUTO_ANALYSE_CELLS = 40000   # stop analysing on every stroke above this size

LIVE_FILL = "#8a5a34"
DEAD_FILL = "#f4f1ec"
GRID_LINE = "#d8d2c8"


# ------------------------------------------------------------ pure helpers

def resize_grid(grid, rows, cols):
    """Change the domain, keeping the drawing centred and cropping if smaller."""
    grid = np.asarray(grid)
    out = np.zeros((rows, cols), dtype=int)

    src_r0 = max(0, (grid.shape[0] - rows) // 2)
    src_c0 = max(0, (grid.shape[1] - cols) // 2)
    take_r = min(rows, grid.shape[0])
    take_c = min(cols, grid.shape[1])
    dst_r0 = max(0, (rows - grid.shape[0]) // 2)
    dst_c0 = max(0, (cols - grid.shape[1]) // 2)

    out[dst_r0:dst_r0 + take_r, dst_c0:dst_c0 + take_c] = \
        grid[src_r0:src_r0 + take_r, src_c0:src_c0 + take_c]
    return out


def analyse(grid, frames, unit=10.0, cell_half=3.575, base_z=-4.0):
    """What this drawing would print as: size, part count, and the warnings."""
    grid = np.asarray(grid)
    if grid.sum() == 0:
        return {"empty": True}

    rec = life_run(grid, frames)
    plan = plan_structure(rec)
    if not plan["cells"]:
        return {"empty": True}

    conn = check_connectivity(plan)
    cells = np.array(plan["cells"], dtype=float)
    xyz = cells * unit
    lo2, hi2 = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
    centre = (lo2 + hi2) / 2
    reach = float(np.linalg.norm(xyz[:, :2] - centre, axis=1).max())

    clipped = bool(rec[:, :2, :].any() or rec[:, -2:, :].any()
                   or rec[:, :, :2].any() or rec[:, :, -2:].any())

    return {
        "empty": False,
        "cells": len(plan["cells"]),
        "rungs": len(plan["rungs"]),
        "height": float(xyz[:, 2].max() + cell_half - base_z),
        "width": (reach + cell_half) * 2,
        "plate": (reach + cell_half + 2.0) * 2,
        "pieces": len(conn["components"]),
        "final_pop": int(rec[-1].sum()),
        "died": bool(rec[-1].sum() == 0),
        "clipped": clipped,
    }


# ------------------------------------------------------------------- the app

class Designer(tk.Tk):
    def __init__(self, grid=None, rows=49, cols=49, frames=22):
        super().__init__()
        self.title("Conway 3D designer")

        if grid is None:
            grid = np.zeros((rows, cols), dtype=int)
        self.cells = np.asarray(grid).astype(int)
        self.rect_ids = {}
        self.drag_value = None
        self.jobs = queue.Queue()
        self.busy = False

        self._build_ui(frames)
        self._redraw_all()
        self.after(120, self._poll_jobs)
        self._analyse_later()

    # ---------------------------------------------------------------- layout
    def _build_ui(self, frames):
        top = ttk.Frame(self, padding=6)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Label(top, text="rows").pack(side="left")
        self.v_rows = tk.IntVar(value=self.cells.shape[0])
        ttk.Spinbox(top, from_=3, to=400, width=5, textvariable=self.v_rows
                    ).pack(side="left", padx=(2, 8))
        ttk.Label(top, text="cols").pack(side="left")
        self.v_cols = tk.IntVar(value=self.cells.shape[1])
        ttk.Spinbox(top, from_=3, to=400, width=5, textvariable=self.v_cols
                    ).pack(side="left", padx=(2, 8))
        ttk.Button(top, text="Resize domain", command=self.do_resize).pack(side="left")
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(top, text="Clear", command=self.do_clear).pack(side="left")
        ttk.Button(top, text="Open…", command=self.do_open).pack(side="left", padx=4)
        ttk.Button(top, text="Save as…", command=self.do_save).pack(side="left")

        self.canvas = tk.Canvas(self, background=DEAD_FILL, highlightthickness=1,
                                highlightbackground="#999")
        self.canvas.grid(row=1, column=0, padx=(8, 4), pady=4)
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_erase)
        self.canvas.bind("<B3-Motion>", self.on_erase)
        self.canvas.bind("<ButtonRelease-3>", self.on_release)

        side = ttk.Frame(self, padding=6)
        side.grid(row=1, column=1, sticky="n")

        ttk.Label(side, text="generations").grid(row=0, column=0, sticky="w")
        self.v_frames = tk.IntVar(value=frames)
        sp = ttk.Spinbox(side, from_=2, to=120, width=6, textvariable=self.v_frames,
                         command=self._analyse_later)
        sp.grid(row=0, column=1, sticky="e")
        sp.bind("<Return>", lambda e: self._analyse_later())

        ttk.Label(side, text="parts").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.v_version = tk.StringVar(value="version3")
        ttk.Combobox(side, width=9, state="readonly", textvariable=self.v_version,
                     values=["version1", "version2", "version3"]
                     ).grid(row=1, column=1, sticky="e", pady=(4, 0))

        self.v_base = tk.BooleanVar(value=True)
        self.v_center = tk.BooleanVar(value=True)
        self.v_binary = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="circular base", variable=self.v_base
                        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(side, text="centre on origin", variable=self.v_center
                        ).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(side, text="binary STL", variable=self.v_binary
                        ).grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Button(side, text="Preview 3D", command=self.do_preview
                   ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 2))
        self.btn_build = ttk.Button(side, text="Generate STL…", command=self.do_build)
        self.btn_build.grid(row=6, column=0, columnspan=2, sticky="ew")

        self.info = tk.Text(side, width=30, height=13, relief="flat", wrap="word",
                            background="#f7f7f5", font=("TkDefaultFont", 9))
        self.info.grid(row=7, column=0, columnspan=2, pady=(10, 0))
        self.info.configure(state="disabled")

        self.status = ttk.Label(self, text="", anchor="w", padding=(8, 3))
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew")

        self.bind("<Control-s>", lambda e: self.do_save())
        self.bind("<Control-o>", lambda e: self.do_open())

    # ---------------------------------------------------------------- canvas
    @property
    def cell_px(self):
        rows, cols = self.cells.shape
        return max(MIN_CELL_PX, min(MAX_CELL_PX, MAX_CANVAS // max(rows, cols)))

    def _redraw_all(self):
        self.canvas.delete("all")
        self.rect_ids.clear()
        rows, cols = self.cells.shape
        px = self.cell_px
        self.canvas.configure(width=cols * px, height=rows * px)
        if px >= 6:
            for c in range(cols + 1):
                self.canvas.create_line(c * px, 0, c * px, rows * px, fill=GRID_LINE)
            for r in range(rows + 1):
                self.canvas.create_line(0, r * px, cols * px, r * px, fill=GRID_LINE)
        for r, c in zip(*np.nonzero(self.cells)):
            self._paint(int(r), int(c), 1)

    def _paint(self, r, c, value):
        """Set one cell and keep only the live ones as canvas items."""
        if self.cells[r, c] == value and (r, c) in self.rect_ids or \
           self.cells[r, c] == value and value == 0:
            return False
        self.cells[r, c] = value
        px = self.cell_px
        if value:
            if (r, c) not in self.rect_ids:
                self.rect_ids[(r, c)] = self.canvas.create_rectangle(
                    c * px + 1, r * px + 1, (c + 1) * px, (r + 1) * px,
                    fill=LIVE_FILL, width=0)
        else:
            item = self.rect_ids.pop((r, c), None)
            if item:
                self.canvas.delete(item)
        return True

    def _cell_at(self, event):
        px = self.cell_px
        r, c = event.y // px, event.x // px
        if 0 <= r < self.cells.shape[0] and 0 <= c < self.cells.shape[1]:
            return int(r), int(c)
        return None

    def on_press(self, event):
        pos = self._cell_at(event)
        if not pos:
            return
        self.drag_value = 0 if self.cells[pos] else 1
        self._paint(*pos, self.drag_value)

    def on_drag(self, event):
        pos = self._cell_at(event)
        if pos and self.drag_value is not None:
            self._paint(*pos, self.drag_value)

    def on_erase(self, event):
        pos = self._cell_at(event)
        if pos:
            self.drag_value = 0
            self._paint(*pos, 0)

    def on_release(self, _event):
        self.drag_value = None
        self._analyse_later()

    # --------------------------------------------------------------- actions
    def do_resize(self):
        try:
            rows, cols = int(self.v_rows.get()), int(self.v_cols.get())
        except (tk.TclError, ValueError):
            return
        self.cells = resize_grid(self.cells, rows, cols)
        self._redraw_all()
        self._analyse_later()

    def do_clear(self):
        self.cells[:] = 0
        self._redraw_all()
        self._analyse_later()

    def do_open(self):
        path = filedialog.askopenfilename(
            title="Open pattern",
            filetypes=[("Pattern files", "*.txt *.cells *.csv *.npy *.npz"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            grid = init_grids.load_pattern(path)
        except Exception as exc:
            messagebox.showerror("Could not open", f"{path}\n\n{exc}")
            return
        self.cells = grid.astype(int)
        self.v_rows.set(self.cells.shape[0])
        self.v_cols.set(self.cells.shape[1])
        self._redraw_all()
        self._analyse_later()
        self.status.configure(text=f"opened {os.path.basename(path)}")

    def do_save(self):
        path = filedialog.asksaveasfilename(
            title="Save pattern", defaultextension=".txt",
            filetypes=[("Pattern text", "*.txt"), ("NumPy array", "*.npy")])
        if not path:
            return
        try:
            init_grids.save_pattern(self.cells, path,
                                    comment="drawn with designer.py")
        except Exception as exc:
            messagebox.showerror("Could not save", f"{path}\n\n{exc}")
            return
        self.status.configure(text=f"saved {os.path.basename(path)}")

    def do_preview(self):
        if self.cells.sum() == 0:
            messagebox.showinfo("Nothing to preview", "Draw some live cells first.")
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from preview import draw, part_extent

        cell_half, base_z = part_extent(f"./model_stls/{self.v_version.get()}")
        rec = life_run(self.cells, int(self.v_frames.get()))

        win = tk.Toplevel(self)
        win.title("Preview")
        fig = Figure(figsize=(5.2, 6.4), dpi=100)
        ax = fig.add_subplot(projection="3d")
        info = draw(ax, rec, 10.0, title="", cell_half=cell_half, base_z=base_z,
                    base_radius="auto" if self.v_base.get() else None)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=win).get_tk_widget().pack(fill="both", expand=True)
        if info:
            ttk.Label(win, padding=6,
                      text=f"{info['cells']} cells, {info['rungs']} rungs, "
                           f"{info['height']:.0f}mm tall").pack()

    def do_build(self):
        if self.busy:
            return
        if self.cells.sum() == 0:
            messagebox.showinfo("Nothing to build", "Draw some live cells first.")
            return
        path = filedialog.asksaveasfilename(
            title="Write STL", defaultextension=".stl",
            initialdir="output_stls" if os.path.isdir("output_stls") else ".",
            filetypes=[("STL", "*.stl")])
        if not path:
            return

        self.busy = True
        self.btn_build.configure(state="disabled")
        self.status.configure(text="building… this can take a minute")
        args = (self.cells.copy(), int(self.v_frames.get()), self.v_version.get(),
                self.v_base.get(), self.v_center.get(), self.v_binary.get(), path)
        threading.Thread(target=self._build_worker, args=args, daemon=True).start()

    def _build_worker(self, grid, frames, version, want_base, want_center,
                      binary, path):
        try:
            blocks = load_building_blocks(f"./model_stls/{version}")
            model = build_model(life_run(grid, frames), blocks, unit=10)
            meshes = list(model["meshes"])
            if want_base:
                plate, _, _, _ = make_circular_base(model, blocks)
                meshes.append(plate)
            if want_center:
                pts = np.concatenate([m.points.reshape(-1, 3) for m in meshes])
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                for m in meshes:
                    m.x -= (lo[0] + hi[0]) / 2
                    m.y -= (lo[1] + hi[1]) / 2
            combined = save_stl(meshes, path, ascii_mode=not binary)
            self.jobs.put(("built", path, len(combined.data),
                           len(model["cells"]), len(model["connection_lines"])))
        except Exception as exc:                      # report, never die silently
            self.jobs.put(("failed", str(exc)))

    def _poll_jobs(self):
        try:
            while True:
                job = self.jobs.get_nowait()
                if job[0] == "built":
                    _, path, tris, cells, rungs = job
                    mb = os.path.getsize(path) / 1e6
                    self.status.configure(
                        text=f"wrote {os.path.basename(path)} — {cells} cells, "
                             f"{rungs} rungs, {tris} triangles, {mb:.0f} MB")
                else:
                    messagebox.showerror("Build failed", job[1])
                    self.status.configure(text="build failed")
                self.busy = False
                self.btn_build.configure(state="normal")
        except queue.Empty:
            pass
        self.after(120, self._poll_jobs)

    # -------------------------------------------------------------- analysis
    def _analyse_later(self):
        if getattr(self, "_pending", None):
            self.after_cancel(self._pending)
        self._pending = self.after(250, self._analyse_now)

    def _analyse_now(self):
        self._pending = None
        rows, cols = self.cells.shape
        if rows * cols > AUTO_ANALYSE_CELLS:
            self._show_info(f"{int(self.cells.sum())} live cells\n\n"
                            f"domain {rows}x{cols} is large, so the summary is "
                            f"not updated automatically.")
            return
        try:
            a = analyse(self.cells, int(self.v_frames.get()))
        except (tk.TclError, ValueError):
            return

        if a.get("empty"):
            self._show_info(f"{int(self.cells.sum())} live cells drawn\n\n"
                            "Nothing survives to build.")
            return

        lines = [f"domain {rows} x {cols}",
                 f"drawn: {int(self.cells.sum())} live cells",
                 "",
                 f"parts:  {a['cells']} cells, {a['rungs']} rungs",
                 f"height: {a['height']:.0f} mm",
                 f"width:  {a['width']:.0f} mm",
                 f"plate:  {a['plate']:.0f} mm across",
                 f"pieces: {a['pieces']}"]
        warn = []
        if a["clipped"]:
            warn.append("The pattern reaches the edge of the domain, so it is "
                        "being cut off and is no longer true Life. Enlarge the "
                        "domain.")
        if a["died"]:
            warn.append("Everything is dead by the last generation.")
        if a["pieces"] > 1 and not self.v_base.get():
            warn.append(f"{a['pieces']} separate pieces. The circular base "
                        "would join them.")
        if warn:
            lines += ["", "— " + "\n— ".join(warn)]
        self._show_info("\n".join(lines))

    def _show_info(self, text):
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", text)
        self.info.configure(state="disabled")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load", help="pattern file to open on start")
    p.add_argument("--pattern", help="named pattern from init_grids.py")
    p.add_argument("--rows", type=int, default=49)
    p.add_argument("--cols", type=int, default=49)
    p.add_argument("--frames", type=int, default=22)
    args = p.parse_args()

    grid = None
    if args.load:
        grid = init_grids.load_pattern(args.load)
    elif args.pattern:
        grid = np.asarray(getattr(init_grids, args.pattern))

    Designer(grid=grid, rows=args.rows, cols=args.cols, frames=args.frames).mainloop()


if __name__ == "__main__":
    main()
