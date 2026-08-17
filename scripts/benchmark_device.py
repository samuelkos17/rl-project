"""Measure environment steps per second on CPU vs CUDA for our actual loop.

Run:  python scripts/benchmark_device.py
"""

import time

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
    print(f"torch {torch.__version__}")
    cpu = run("cpu")
    print(f"cpu   {cpu:8.1f} steps/s   -> 400k steps in {400_000 / cpu / 60:.1f} min")
    if torch.cuda.is_available():
        cuda = run("cuda")
        print(f"cuda  {cuda:8.1f} steps/s   -> 400k steps in {400_000 / cuda / 60:.1f} min")
        print(f"\nfaster device: {'cuda' if cuda > cpu else 'cpu'}")
    else:
        print("cuda  UNAVAILABLE -- this build cannot measure the GPU")
