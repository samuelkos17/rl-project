# Task 4 — NoisyNets

The only task where you touch a file Samuel owns. He ships `NoisyLinear` as a
placeholder that behaves like a normal layer; you replace the body. **Do not
change the class name, the constructor signature, or the method names** — his
`QNetwork` calls them.

**Files:**
- Modify: `src/rlx/networks.py` (the `NoisyLinear` class only)
- Create: `src/rlx/exploration/noisy.py`
- Test: `tests/test_exploration/test_noisy.py`

**Interfaces:**
- Consumes: `QNetwork`, `NoisyLinear` placeholder (Samuel task 4).
- Produces: working `NoisyLinear`, and `NoisyExplorer(cfg, rng)` with
  `uses_noisy_net = True`.

**Config fields you read:** `noisy_sigma0` (0.5).

---

## What NoisyNets is

The other three strategies add randomness to the **action choice**. NoisyNets
adds randomness to the **network's weights** instead.

Every weight becomes `mu + sigma * epsilon`, where `mu` is the usual learned
weight, `sigma` is a second learned parameter saying "how unsure am I about this
weight", and `epsilon` is fresh random noise resampled on each forward pass.

The interesting part is that **`sigma` is learned by gradient descent like any
other parameter**. The network works out for itself where it still benefits from
randomness and where it has settled — instead of us imposing a decay schedule we
guessed. That is the whole pitch: exploration that tunes itself.

Two consequences:

- Action selection is **purely greedy**. `act()` is just `argmax`. All the
  exploration already happened inside the network.
- Noise must be **off during evaluation** (use the `mu` weights only), otherwise
  reported scores include random jitter. Samuel's `evaluate()` calls
  `set_noise_enabled(False)`; your job is to make that flag actually do something.

## Factorised Gaussian noise

The paper offers two variants; we use the **factorised** one because it is
cheaper and standard. Instead of drawing one noise value per weight (which for a
1024x64 layer is 65,536 draws every forward pass), draw one vector per input and
one per output, and combine them:

```
epsilon_w[i, j] = f(epsilon_in[j]) * f(epsilon_out[i])
epsilon_b[i]    = f(epsilon_out[i])
where f(x) = sign(x) * sqrt(|x|)
```

That is 1088 draws instead of 65,536 for the same layer.

Initialisation from the paper:
- `mu` uniform in `[-1/sqrt(in_features), +1/sqrt(in_features)]`
- `sigma` constant at `sigma0 / sqrt(in_features)`

---

- [x] **Step 1: Write the failing tests**

Create `tests/test_exploration/test_noisy.py`:

```python
import numpy as np
import torch

from rlx.exploration.noisy import NoisyExplorer
from rlx.networks import NoisyLinear, QNetwork


def test_noisy_layer_has_learnable_mu_and_sigma():
    layer = NoisyLinear(8, 4, sigma0=0.5)
    names = {n for n, p in layer.named_parameters() if p.requires_grad}
    assert {"weight_mu", "weight_sigma", "bias_mu", "bias_sigma"} <= names


def test_output_changes_when_noise_is_resampled():
    torch.manual_seed(0)
    layer = NoisyLinear(8, 4, sigma0=0.5)
    x = torch.ones(1, 8)
    layer.reset_noise()
    a = layer(x).detach().clone()
    layer.reset_noise()
    b = layer(x).detach().clone()
    assert not torch.allclose(a, b)


def test_output_is_deterministic_when_noise_is_disabled():
    torch.manual_seed(0)
    layer = NoisyLinear(8, 4, sigma0=0.5)
    layer.noise_enabled = False
    x = torch.ones(1, 8)
    layer.reset_noise()
    a = layer(x).detach().clone()
    layer.reset_noise()
    assert torch.allclose(a, layer(x).detach())


def test_disabled_noise_uses_the_mu_weights_exactly():
    layer = NoisyLinear(8, 4, sigma0=0.5)
    layer.noise_enabled = False
    x = torch.ones(1, 8)
    expected = torch.nn.functional.linear(x, layer.weight_mu, layer.bias_mu)
    assert torch.allclose(layer(x), expected)


def test_sigma_receives_gradients():
    """If sigma never learns, this is not NoisyNets."""
    layer = NoisyLinear(8, 4, sigma0=0.5)
    layer.reset_noise()
    layer(torch.ones(1, 8)).sum().backward()
    assert layer.weight_sigma.grad is not None
    assert layer.weight_sigma.grad.abs().sum() > 0


def test_qnetwork_with_noisy_head_produces_varying_outputs():
    torch.manual_seed(0)
    net = QNetwork(n_actions=7, noisy=True, sigma0=0.5)
    x = torch.rand(1, 3, 7, 7)
    net.reset_noise()
    a = net(x).detach().clone()
    net.reset_noise()
    assert not torch.allclose(a, net(x).detach())


def test_qnetwork_with_noise_disabled_is_deterministic():
    net = QNetwork(n_actions=7, noisy=True, sigma0=0.5)
    net.set_noise_enabled(False)
    x = torch.rand(1, 3, 7, 7)
    net.reset_noise()
    a = net(x).detach().clone()
    net.reset_noise()
    assert torch.allclose(a, net(x).detach())


def test_explorer_declares_it_needs_a_noisy_network(cfg, rng):
    assert NoisyExplorer(cfg, rng).uses_noisy_net is True


def test_explorer_acts_purely_greedily(cfg, rng, q_values, key):
    e = NoisyExplorer(cfg, rng)
    assert all(e.act(q_values, key, s) == 3 for s in (0, 5000, 10_000))


def test_explorer_adds_no_intrinsic_bonus(cfg, rng, key):
    assert NoisyExplorer(cfg, rng).intrinsic_bonus(key) == 0.0
```

- [x] **Step 2: Run and watch them fail**

```bash
pytest tests/test_exploration/test_noisy.py -v
```

Expected: failures on `weight_mu` not existing (the placeholder is a plain
`nn.Linear`) plus a `ModuleNotFoundError` for `rlx.exploration.noisy`.

- [x] **Step 3: Replace `NoisyLinear` in `src/rlx/networks.py`**

Replace the placeholder class body. Leave the rest of the file alone.

```python
class NoisyLinear(nn.Module):
    """Linear layer with learned factorised Gaussian weight noise.

    weight = weight_mu + weight_sigma * epsilon, with epsilon resampled by
    reset_noise(). weight_sigma is learned, so the network decides for itself
    how much randomness each weight still needs.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.noise_enabled = True

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers, not parameters: noise is resampled, never learned.
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        bound = 1.0 / np.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma0 * bound)
        nn.init.constant_(self.bias_sigma, sigma0 * bound)

        self.reset_noise()

    @staticmethod
    def _scaled_noise(size: int, device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        device = self.weight_mu.device
        eps_in = self._scaled_noise(self.in_features, device)
        eps_out = self._scaled_noise(self.out_features, device)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.noise_enabled:
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return nn.functional.linear(x, weight, bias)
```

Add `import numpy as np` at the top of `networks.py` if it is not already there.

Note `NoisyLinear` now subclasses `nn.Module`, not `nn.Linear`. `QNetwork`'s
`isinstance(m, NoisyLinear)` checks still work, and the plain-`nn.Linear` branch
is unaffected.

- [x] **Step 4: Write `src/rlx/exploration/noisy.py`**

```python
"""NoisyNets: exploration comes from learned weight noise inside the network,
so action selection is purely greedy."""

import numpy as np

from rlx.config import RunConfig
from rlx.exploration.base import Explorer


class NoisyExplorer(Explorer):
    uses_noisy_net = True

    def __init__(self, cfg: RunConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def act(self, q_values: np.ndarray, count_key, step: int) -> int:
        # Greedy on purpose. The noise already happened inside the network.
        return int(np.argmax(q_values))
```

That is the whole class. Rule 6: less code is better. All the machinery lives in
`NoisyLinear`; the explorer only declares that it needs one.

**One documented simplification.** The original paper resamples noise for the
online *and* target networks on every gradient update. Our training loop only
calls `reset_noise()` before acting, so a batch update reuses the noise drawn for
that step, and the target network keeps whatever noise it was initialised with.

This is a deliberate simplification, not an oversight: it keeps the training loop
identical across all four strategies, which is the controlled-comparison property
the whole project depends on. Exploration still works, because the noise driving
action selection is resampled every step — which is what actually explores.

**Put this in the report's limitations section.** Do not silently deviate from a
cited paper; naming the deviation and the reason is what makes it defensible.

- [x] **Step 5: Run the tests**

```bash
pytest tests/test_exploration/test_noisy.py -v
```

Expected: 10 passed.

If `test_sigma_receives_gradients` fails, `weight_sigma` was registered as a
buffer instead of a `nn.Parameter` — that would mean the noise level never
learns, which is the one thing that makes NoisyNets different from adding random
jitter.

- [x] **Step 6: Run the whole exploration suite**

```bash
pytest tests/test_exploration/ -v
```

All four strategies, everything green. Then check all four build through the
factory:

```bash
python -c "
import numpy as np
from rlx.config import RunConfig
from rlx.exploration import STRATEGIES, make_explorer
for s in STRATEGIES:
    cfg = RunConfig(env_id='Empty-5', strategy=s, seed=0)
    e = make_explorer(s, cfg, np.random.default_rng(0))
    print(f'{s:<15} {type(e).__name__:<15} uses_noisy_net={e.uses_noisy_net}')
"
```

Expected: four lines, `uses_noisy_net=True` on `noisy` only.

- [x] **Step 7: Tell Samuel**

His `test_every_strategy_runs_end_to_end` was marked `xfail` while your modules
did not exist. Tell him to remove the marker and run it.

- [ ] **Step 8: Log and commit**

Append to `docs/decision_log.md`:

```markdown
## 2026-08-19 — NoisyNets implemented

**Status:** Active

**What changed:** Added the fourth and last strategy. The other three add
randomness to *which action* the agent picks. This one adds randomness to the
network's *own weights* instead, and then chooses actions with no randomness at
all.

**The interesting bit:** the amount of noise is a learned parameter. The network
works out for itself which parts of it still benefit from randomness, instead of
us picking a decay schedule by hand like we did for the other strategies. That is
the main argument for it in the original paper.

**What it means for the results:** We switch the noise off when measuring
performance, so reported scores use the network's average weights. If we forgot
that, every score would have random jitter in it. There is a test that fails if
someone breaks it.

**One implementation note:** we use the "factorised" noise variant, which draws
one random number per input and per output and multiplies them, instead of one
per weight. For our biggest layer that is about 1,000 random numbers per step
instead of 65,000 — same idea, far cheaper.
```

```bash
git add src/rlx/networks.py src/rlx/exploration/noisy.py tests/test_exploration/test_noisy.py docs/decision_log.md
git commit -m "feat: NoisyNets exploration with factorised Gaussian noise"
```
