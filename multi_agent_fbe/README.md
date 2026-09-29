# Multi-agent greedy frontier-based exploration (32x32 grid)

`greedy_frontier_grid.py` runs 4 robots doing greedy frontier-based exploration/coverage
on a 32x32 grid. It ports the decision rule of `BaseExplorer` from
[naokiyokoyama/frontier_exploration](https://github.com/naokiyokoyama/frontier_exploration)
(cloned as a submodule at `../external/frontier_exploration`): each robot goes to the
frontier with the lowest path cost (`_get_closest_waypoint` / `_astar_search`). The
original is a Habitat sensor that needs habitat-sim and a 3D scene, so the logic is
re-implemented on a grid instead of imported.

```
pip install -r requirements.txt
python greedy_frontier_grid.py            # --size 32 --robots 4 --obstacles 0.2 --seed 0
```

| Experiment | Free cells | Steps to 100% coverage | Total moves | Redundant moves |
|---|---|---|---|---|
| 1: open grid | 1024 | 255 | 1020 | 0 |
| 2: 20% random obstacles (seed 0) | 819 | 445 | 1777 | 962 |

Plots: `results/exp1_open_grid.png`, `results/exp2_obstacles.png`; metrics: `results/summary.json`.
