Hi you three,

 

Thank you for the proposal. The controlled setup (one DQN, swapping only the exploration module) is the right instinct, and the part I like most is that you do not stop at a ranking—you want to explain it via state coverage. That coverage-explains-performance angle is your core.

I have the following comments: 

- Define "state coverage" carefully. MiniGrid observations are partial (a 7×7×3 local view), while the true state is in the form of (x, y, direction). Coverage over the underlying (x, y[, dir]) is the meaningful quantity, but note that using it is privileged information the agent does not see. This is fine for analysis, just be explicit about it. 
- You can consider turn the central hypothesis (higher early coverage can lead to better later performance) into a concrete, quantitative test e.g. correlate an early-training coverage measure, e.g., AUC over the first k steps) against final return across strategies and environments. In your current version, it is only stated informally.
- Implementation notes: the count-based bonus can use cheap tabular state counts on MiniGrid; NoisyNets is a clean drop-in; Boltzmann needs a stated temperature schedule. Say how each intrinsic bonus interacts with the evaluation return (evaluate greedily on extrinsic reward only).
- DQN and all four exploration modules are standard, well-documented components. A few ways to extend the same idea:
  - Make the difficulty continuous rather than 3 fixed points: parametrize MultiRoom (room count) and DoorKey (grid size) into 5-6 instances. I imagine this could be cheap to add, and in the end you may show your central coverage-vs-performance claim with an actual curve.
  - Decompose "coverage" into raw vs. task-relevant (states on/near the optimal path, adjacent to the key/door/goal, e.g. via distance-to-goal) and test which one actually predicts performance. Some of them reuse the visitation logs you're already collecting and therefore easy to compute
  - Once you have difficult levels, you can check rank stability: does the strategy that wins on Empty still win on MultiRoom, or will the rank be different?
- The following are minor:
  - Commit the number of seeds (≥5) and the specific MiniGrid variants/sizes (DoorKey-?, MultiRoom-N?) — difficulty and reproducibility depend on it. Good that you already plan to use rliable.
  - After raising a hypothesis, it could be good to talk about what result would confirm your hypothesis.
  - Specify the DQN variant (vanilla/double/dueling) and keep it fixed across strategies.
