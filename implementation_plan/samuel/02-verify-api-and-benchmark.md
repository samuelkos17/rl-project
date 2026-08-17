# Task 2 — Verify the MiniGrid API, then benchmark CPU vs GPU

Two questions this task answers, both of which the whole plan currently only
*assumes*:

1. Do the MiniGrid classes and attributes we planned around actually exist with
   those names?
2. Is CPU or GPU faster for our tiny network, and how many steps can we afford?

**Do not skip this because it produces no library code.** Every later task builds
on the answers.

**Files:**
- Create: `scripts/verify_api.py`
- Create: `scripts/benchmark_device.py`

**Interfaces:**
- Consumes: `RunConfig` from Task 1.
- Produces: verified attribute names for Task 3; final `total_steps` and `device`
  values for `configs/main.yaml`.

---

- [x] **Step 1: Write the API verification script**

Create `scripts/verify_api.py`. This asserts every assumption in spec §3.

```python
"""Verify every MiniGrid API assumption in the spec before anything depends on it.

Run:  python scripts/verify_api.py
"""

import numpy as np
from minigrid.envs import DoorKeyEnv, EmptyEnv, MultiRoomEnv
from minigrid.wrappers import ImgObsWrapper

CASES = [
    ("Empty-5", lambda: EmptyEnv(size=5)),
    ("Empty-8", lambda: EmptyEnv(size=8)),
    ("Empty-16", lambda: EmptyEnv(size=16)),
    ("DoorKey-5", lambda: DoorKeyEnv(size=5)),
    ("DoorKey-6", lambda: DoorKeyEnv(size=6)),
    ("DoorKey-7", lambda: DoorKeyEnv(size=7)),
    ("DoorKey-8", lambda: DoorKeyEnv(size=8)),
    ("DoorKey-10", lambda: DoorKeyEnv(size=10)),
    ("MultiRoom-N2", lambda: MultiRoomEnv(minNumRooms=2, maxNumRooms=2, maxRoomSize=6)),
    ("MultiRoom-N3", lambda: MultiRoomEnv(minNumRooms=3, maxNumRooms=3, maxRoomSize=6)),
    ("MultiRoom-N4", lambda: MultiRoomEnv(minNumRooms=4, maxNumRooms=4, maxRoomSize=6)),
    ("MultiRoom-N5", lambda: MultiRoomEnv(minNumRooms=5, maxNumRooms=5, maxRoomSize=6)),
    ("MultiRoom-N6", lambda: MultiRoomEnv(minNumRooms=6, maxNumRooms=6, maxRoomSize=6)),
]


def check(name, factory):
    env = ImgObsWrapper(factory())
    obs, info = env.reset(seed=0)

    assert obs.shape == (7, 7, 3), f"{name}: obs shape {obs.shape}, expected (7,7,3)"
    assert env.action_space.n == 7, f"{name}: {env.action_space.n} actions, expected 7"

    u = env.unwrapped
    assert hasattr(u, "agent_pos"), f"{name}: no agent_pos"
    assert hasattr(u, "agent_dir"), f"{name}: no agent_dir"
    assert hasattr(u, "grid"), f"{name}: no grid"
    assert u.agent_dir in (0, 1, 2, 3), f"{name}: agent_dir={u.agent_dir}"
    assert len(u.agent_pos) == 2, f"{name}: agent_pos={u.agent_pos}"

    obs2, r, term, trunc, info = env.step(2)
    assert obs2.shape == (7, 7, 3)
    assert isinstance(float(r), float)

    # Layout pinning: resetting with the same seed must reproduce the same maze.
    a, _ = env.reset(seed=7)
    layout_a = _layout_fingerprint(env)
    b, _ = env.reset(seed=7)
    layout_b = _layout_fingerprint(env)
    assert layout_a == layout_b, f"{name}: reset(seed=7) is not reproducible"

    c, _ = env.reset(seed=8)
    layout_c = _layout_fingerprint(env)
    varies = layout_a != layout_c

    print(f"OK  {name:<14} grid={u.width}x{u.height}  "
          f"max_steps={u.max_steps}  layout_varies_by_seed={varies}")
    env.close()


def _layout_fingerprint(env) -> str:
    u = env.unwrapped
    cells = []
    for y in range(u.height):
        for x in range(u.width):
            c = u.grid.get(x, y)
            cells.append("." if c is None else c.type[0])
    return "".join(cells)


if __name__ == "__main__":
    for name, factory in CASES:
        check(name, factory)
    print("\nAll API assumptions verified.")
```

- [x] **Step 2: Run it**

```bash
python scripts/verify_api.py
```

Expected: 13 `OK` lines and `All API assumptions verified.`

**If anything fails — a class name, a keyword argument, an attribute — stop.**
Find the real API in the installed package (`python -c "import minigrid.envs as e; help(e.DoorKeyEnv)"`),
fix `scripts/verify_api.py` and the spec's §3 assumption list to match reality,
and write a `docs/decision_log.md` entry recording what the real API turned out
to be. Do not guess a second time; look it up.

**Record the printed `max_steps` for each instance** — Task 3 needs it and
Daniel's coverage denominator sanity check uses it.

- [x] **Step 3: Check `layout_varies_by_seed`**

Every `DoorKey-*` and `MultiRoom-*` row should print `True`. `Empty-*` may print
`False`, which is fine — Empty has a fixed layout by construction.

If a `DoorKey` or `MultiRoom` row prints `False`, the layout is not seed-dependent
after all and the per-run layout pinning in the spec needs revisiting. Report it
before continuing.

- [x] **Step 4: Write the device benchmark**

Create `scripts/benchmark_device.py`. It measures the real loop — environment
step plus gradient update — not just matrix multiplication.

```python
"""Measure environment steps per second on CPU vs CUDA for our actual loop.

Run:  python scripts/benchmark_device.py
"""

import time

import numpy as np
import torch
import torch.nn as nn
from minigrid.envs import DoorKeyEnv
from minigrid.wrappers import ImgObsWrapper

STEPS = 3_000
BATCH = 32
TRAIN_FREQ = 4


def build_net():
    return nn.Sequential(
        nn.Conv2d(3, 16, 2), nn.ReLU(),
        nn.Conv2d(16, 32, 2), nn.ReLU(),
        nn.Conv2d(32, 64, 2), nn.ReLU(),
        nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 7),
    )


def run(device: str) -> float:
    env = ImgObsWrapper(DoorKeyEnv(size=8))
    obs, _ = env.reset(seed=0)

    net = build_net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    fake = torch.rand(BATCH, 3, 7, 7, device=device)
    target = torch.rand(BATCH, 7, device=device)

    start = time.perf_counter()
    for t in range(STEPS):
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        x = (x / 10.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            net(x)
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset(seed=0)
        if t % TRAIN_FREQ == 0:
            loss = nn.functional.smooth_l1_loss(net(fake), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    env.close()
    return STEPS / elapsed


if __name__ == "__main__":
    cpu = run("cpu")
    print(f"cpu   {cpu:8.1f} steps/s   -> 400k steps in {400_000 / cpu / 60:.1f} min")
    if torch.cuda.is_available():
        cuda = run("cuda")
        print(f"cuda  {cuda:8.1f} steps/s   -> 400k steps in {400_000 / cuda / 60:.1f} min")
        print(f"\nfaster device: {'cuda' if cuda > cpu else 'cpu'}")
    else:
        print("cuda  unavailable")
```

- [x] **Step 5: Run the benchmark**

```bash
python scripts/benchmark_device.py
```

Read the actual numbers. Do not assume the answer — the whole point of this step
is that we do not know it.

- [x] **Step 6: Decide `total_steps` and `device` from the measurement**

Budget arithmetic. You have 260 runs, 3 machines, and roughly 8 parallel workers
per machine, so about 24 concurrent runs. Wall clock is therefore:

```
wall_clock_hours ~= 260 / 24 * (total_steps / steps_per_second) / 3600
```

Note parallel workers contend for cores, so per-worker throughput will be lower
than the single-process benchmark — assume roughly half as a planning figure.

Pick the largest `total_steps` that keeps the full sweep under **6 hours** of
wall clock. Round to a clean number (400k, 300k, or 200k). Do not exceed 400k
even if it fits; more steps is not the goal.

- [x] **Step 7: Update `configs/main.yaml`, and re-check `snapshot_every` with it**

Set `total_steps` and `device` to the measured values. Delete the two
`# provisional` comments.

**Then check the snapshot resolution, because it depends on `total_steps`.** The
early-coverage AUC — the predictor the whole project rests on — is integrated
over the first 20% of training, so the number of snapshots inside that window is
`0.2 * total_steps / snapshot_every`. It must be **at least 8**:

```bash
python -c "
from rlx.config import RunConfig
c = RunConfig(env_id='Empty-5', strategy='epsilon_greedy', seed=0)
n = int(0.2 * c.total_steps / c.snapshot_every)
print(f'total_steps={c.total_steps:,} snapshot_every={c.snapshot_every:,} -> {n} snapshots in the early window')
print('OK' if n >= 8 else 'TOO COARSE: halve snapshot_every in config.py')
"
```

| `total_steps` | required `snapshot_every` |
|---|---|
| 400_000 | 10_000 |
| 300_000 | 5_000 |
| 200_000 | 5_000 |

If it prints `TOO COARSE`, change `snapshot_every` in `src/rlx/config.py` and
tell Daniel — his synthetic generator hard-codes `SNAPSHOT_EVERY` and must match.
Storage is about 1 KB per snapshot, so extra resolution is effectively free; a
4-point trapezoid on the main predictor is not.

- [x] **Step 8: Log both results**

Append to `docs/decision_log.md`, replacing the open "CPU vs GPU" entry's status
with `Resolved`. Write plainly, with the real numbers:

```markdown
## 2026-08-18 — Benchmark result: <cpu|gpu> is faster, budget set to <N> steps

**Status:** Active — resolves the open entry from 2026-08-17

**What changed:** We measured it instead of guessing. CPU did <X> steps per
second, GPU did <Y>. We are using <device>, and each run gets <N> training steps.

**Why:** <one sentence on why that device won — e.g. "the network is so small
that the cost of sending work to the graphics card outweighs the arithmetic",
or "the GPU won anyway, our expectation was wrong">.

**What it means for the results:** The full set of 260 runs should take about
<Z> hours spread over our three PCs. <N> steps is what fits in the time we have.

**Also verified:** all 13 MiniGrid environments build and behave as the spec
assumes. <Note any API that turned out to be different.>
```

If the GPU won, say so plainly. Being wrong and measuring it is the point.

- [ ] **Step 9: Commit**

```bash
git add scripts/ configs/main.yaml docs/decision_log.md
git commit -m "chore: verify MiniGrid API and set step budget from benchmark"
```
