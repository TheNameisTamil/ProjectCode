"""Multi-agent greedy frontier-based exploration and coverage on a 2D grid.

Ports the decision rule of ``BaseExplorer`` from naokiyokoyama/frontier_exploration
(external/frontier_exploration/frontier_exploration/base_explorer.py) to a discrete
grid: each robot repeatedly picks the frontier with the lowest *path* cost from its
current position (``_get_closest_waypoint`` -> ``_astar_search``) and takes one step
along that path. The original class is a Habitat sensor that needs habitat-sim and a
3D scene, so the greedy logic is re-implemented here rather than imported.

Model
-----
* Grid cells are free (0) or obstacle (1). Robots move 4-connected, one cell per step.
* Each robot senses the 8 cells around it; all robots share one map (centralised comms).
* A frontier is a known-free cell that no robot has visited yet, i.e. the boundary
  between covered and uncovered space. Exploration ends when no frontier remains,
  at which point every reachable free cell has been physically visited (100% coverage).
* Coordination: robots plan in order; a frontier targeted by an earlier robot in the
  same step is skipped by later robots, and a robot waits if its next cell is occupied.

Run:  python greedy_frontier_grid.py [--size 32] [--robots 4] [--obstacles 0.2] [--seed 0]
"""

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

UNKNOWN, FREE, OBSTACLE = -1, 0, 1
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLORS = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]


def make_grid(size, obstacle_ratio, starts, rng):
    """Random obstacle grid; start cells are always free.

    Obstacles are dropped on random cells one at a time, skipping any cell that would
    split the free space, so all free cells stay reachable and no robot is walled in.
    """
    grid = np.zeros((size, size), dtype=np.int8)
    n_obs = int(round(obstacle_ratio * size * size))
    candidates = [(r, c) for r in range(size) for c in range(size) if (r, c) not in starts]
    placed = 0
    for i in rng.permutation(len(candidates)):
        if placed == n_obs:
            break
        grid[candidates[i]] = OBSTACLE
        if len(reachable_cells(grid, starts[:1])) == size * size - placed - 1:
            placed += 1
        else:
            grid[candidates[i]] = FREE
    return grid


def reachable_cells(grid, starts):
    """Free cells reachable from any start: the coverage target."""
    seen = set(starts)
    queue = deque(starts)
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if grid[nr, nc] == FREE and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
    return seen


def sense(known, grid, pos):
    r, c = pos
    r0, r1 = max(r - 1, 0), min(r + 2, grid.shape[0])
    c0, c1 = max(c - 1, 0), min(c + 2, grid.shape[1])
    known[r0:r1, c0:c1] = grid[r0:r1, c0:c1]


def closest_frontier_path(known, start, is_target):
    """BFS over known-free cells (uniform-cost A*); returns path to nearest target."""
    parent = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell != start and is_target(cell):
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return path[::-1]
        r, c = cell
        for dr, dc in MOVES:
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < known.shape[0] and 0 <= nxt[1] < known.shape[1]:
                if known[nxt] == FREE and nxt not in parent:
                    parent[nxt] = cell
                    queue.append(nxt)
    return None


def run(grid, starts, max_steps=10_000):
    known = np.full(grid.shape, UNKNOWN, dtype=np.int8)
    visited = np.zeros(grid.shape, dtype=bool)
    owner = np.full(grid.shape, -1, dtype=np.int8)  # robot that first covered each cell
    positions = list(starts)
    paths = [[p] for p in starts]
    target = reachable_cells(grid, starts)

    for i, p in enumerate(positions):
        sense(known, grid, p)
        visited[p] = True
        owner[p] = i
    coverage = [visited.sum() / len(target)]

    step = 0
    while step < max_steps:
        claimed = set()
        any_frontier = False
        for i in range(len(positions)):
            def is_frontier(cell, claimed=claimed):
                return not visited[cell] and cell not in claimed

            path = closest_frontier_path(known, positions[i], is_frontier)
            if path is None:
                # every remaining frontier is claimed: still head for the nearest one
                path = closest_frontier_path(known, positions[i], lambda c: not visited[c])
            if path is None:
                paths[i].append(positions[i])
                continue
            any_frontier = True
            claimed.add(path[-1])
            nxt = path[1]
            if nxt in positions:  # another robot is there: wait this step
                paths[i].append(positions[i])
                continue
            positions[i] = nxt
            paths[i].append(nxt)
            sense(known, grid, nxt)
            if not visited[nxt]:
                visited[nxt] = True
                owner[nxt] = i
        if not any_frontier:
            break
        step += 1
        coverage.append(visited.sum() / len(target))

    return {
        "steps": step,
        "paths": paths,
        "owner": owner,
        "visited": visited,
        "coverage": coverage,
        "target_cells": len(target),
    }


def trim_trailing_waits(path):
    while len(path) > 1 and path[-1] == path[-2]:
        path = path[:-1]
    return path


def plot(grid, starts, result, title, out_path):
    n = len(starts)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    size = grid.shape[0]

    # 1) coverage paths
    ax = axes[0]
    ax.imshow(grid, cmap=ListedColormap(["white", "black"]), origin="upper",
              extent=(-0.5, size - 0.5, size - 0.5, -0.5))
    offsets = np.linspace(-0.18, 0.18, n)
    for i, path in enumerate(result["paths"]):
        path = trim_trailing_waits(path)
        ys = [p[0] + offsets[i] for p in path]
        xs = [p[1] + offsets[i] for p in path]
        ax.plot(xs, ys, "-", color=COLORS[i], lw=1.3, alpha=0.9,
                label=f"Robot {i + 1} ({len(path) - 1} moves)")
        ax.plot(xs[0], ys[0], "o", color=COLORS[i], ms=10, mec="black")
        ax.plot(xs[-1], ys[-1], "X", color=COLORS[i], ms=11, mec="black")
    ax.set_title("Robot coverage paths  (o start, X end)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=9)

    # 2) which robot covered each cell first
    ax = axes[1]
    img = np.ones((size, size, 3))
    for i in range(n):
        img[result["owner"] == i] = matplotlib.colors.to_rgb(COLORS[i])
    img = 0.35 * img + 0.65 * np.ones_like(img)  # pastel
    img[result["owner"] == -1] = (0.85, 0.85, 0.85)  # unreachable / unvisited free
    img[grid == OBSTACLE] = (0, 0, 0)
    ax.imshow(img, origin="upper")
    for i, s in enumerate(starts):
        ax.plot(s[1], s[0], "o", color=COLORS[i], ms=10, mec="black")
    counts = [(result["owner"] == i).sum() for i in range(n)]
    ax.set_title("Cells first covered by each robot\n" +
                 ", ".join(f"R{i + 1}: {c}" for i, c in enumerate(counts)), fontsize=10)

    for a in axes[:2]:
        a.set_xticks(range(0, size, 4))
        a.set_yticks(range(0, size, 4))
        a.set_xticks(np.arange(-0.5, size, 1), minor=True)
        a.set_yticks(np.arange(-0.5, size, 1), minor=True)
        a.grid(which="minor", color="0.85", lw=0.4)
        a.tick_params(which="minor", length=0)

    # 3) coverage over time
    ax = axes[2]
    cov = 100 * np.array(result["coverage"])
    ax.plot(cov, color="tab:blue", lw=2)
    ax.axhline(100, color="grey", ls="--", lw=1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Coverage of reachable free cells (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Coverage vs time: 100% in {result['steps']} steps")
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def summarize(name, grid, result):
    moves = [len(trim_trailing_waits(p)) - 1 for p in result["paths"]]
    return {
        "experiment": name,
        "grid_size": grid.shape[0],
        "obstacle_cells": int((grid == OBSTACLE).sum()),
        "reachable_free_cells": result["target_cells"],
        "unreachable_free_cells": int((grid == FREE).sum()) - result["target_cells"],
        "steps_to_full_coverage": result["steps"],
        "final_coverage_pct": round(100 * result["coverage"][-1], 2),
        "moves_per_robot": moves,
        "total_moves": sum(moves),
        "redundant_moves": sum(moves) - (result["target_cells"] - len(result["paths"])),
        "cells_first_covered_per_robot": [int((result["owner"] == i).sum())
                                          for i in range(len(result["paths"]))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--obstacles", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    s = args.size - 1
    starts = [(0, 0), (0, s), (s, 0), (s, s), (s // 2, 0), (s // 2, s)][: args.robots]
    rng = np.random.default_rng(args.seed)

    experiments = [
        ("exp1_open_grid", 0.0, f"Experiment 1: {args.robots} robots, "
                                f"{args.size}x{args.size} open grid"),
        ("exp2_obstacles", args.obstacles, f"Experiment 2: {args.robots} robots, "
                                           f"{args.size}x{args.size} grid, "
                                           f"{int(args.obstacles * 100)}% random obstacles "
                                           f"(seed {args.seed})"),
    ]
    summaries = []
    for name, ratio, title in experiments:
        grid = make_grid(args.size, ratio, starts, rng)
        result = run(grid, starts)
        plot(grid, starts, result, title, args.out / f"{name}.png")
        summary = summarize(name, grid, result)
        summaries.append(summary)
        print(json.dumps(summary))

    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
