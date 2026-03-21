# ECE-4524-BirdsEye

A real-time autonomous rover navigation system using a top-down 2D simulation.
A Deep Q-Network (DQN) agent learns to drive a car to randomly placed targets
while avoiding screen boundaries, transferring the learned policy from
simulation toward a physical rover.

## Project Structure

```
ECE-4524-BirdsEye/
├── TestRig.py          # Pygame simulation (environment, entities, rendering)
├── car_env.py          # Gym-style wrapper with state flattening and reward shaping
├── dqn_agent.py        # Double DQN network, replay buffer, and agent logic
├── train.py            # Training loop, evaluation modes, and plotting
├── requirements.txt    # Python dependencies
├── dqn_model.pth       # Saved model weights (generated after training)
└── training_plot.png   # Reward-vs-episode plot (generated after training)
```

## File Descriptions

### TestRig.py

The core Pygame simulation. Contains all game entities and physics:

- **`Car`** -- A rectangular vehicle with position, heading angle, and
  forward/backward/rotate movement. Uses pixel-perfect mask collision via
  `pygame.Surface`.
- **`Target`** -- A circle placed randomly on the field. The car must drive
  into it to score.
- **`Obstacle`** -- Randomly shaped barriers (rect, circle, triangle, polygon).
  Currently disabled (`Obstacles = False`) but fully implemented for future use.
- **`Game`** -- Orchestrates the simulation. Provides `step(action)` which
  advances one physics tick and returns `(game_state, terminated, hit_target)`,
  and `getGameState()` which returns the car's corner positions, heading, and
  target location. Supports headless mode via `render_mode` parameter for fast
  training without opening a window.

**Actions:**

| ID | Action        |
|----|---------------|
| 0  | Drive forward |
| 1  | Drive backward|
| 2  | Rotate left   |
| 3  | Rotate right  |
| 4  | No-op         |

Run the simulation manually with WASD controls:

```bash
python TestRig.py
```

Print the screenshot-derived state array each time a new capture is processed:

```bash
python TestRig.py --vision-debug --capture-interval 0.5
```

Print the expanded vision state instead of the compact array:

```bash
python TestRig.py --vision-debug --vision-full-state --capture-interval 0.5
```

### car_env.py

An OpenAI Gym-style wrapper (`CarEnv`) around `Game` that provides standard
`reset()` and `step(action)` methods.

**Responsibilities:**

- **State flattening** -- Converts the `getGameState()` dictionary into a
  normalized 22-element float32 NumPy vector (see State Representation below).
- **Reward computation** -- Applies the dense reward function described below.
- **Episode termination** -- Ends the episode on target hit, boundary/obstacle
  collision, or exceeding `max_steps`.

### dqn_agent.py

All reinforcement learning components:

- **`DQNetwork`** -- A feed-forward PyTorch neural network.
  Architecture: `Linear(22, 256) -> ReLU -> Linear(256, 256) -> ReLU ->
  Linear(256, 128) -> ReLU -> Linear(128, 5)`.
- **`ReplayBuffer`** -- A fixed-capacity deque storing
  `(state, action, reward, next_state, done)` transitions. Supports random
  batch sampling that returns GPU-ready tensors.
- **`DQNAgent`** -- The training agent. Implements Double DQN with soft
  (Polyak) target-network updates, epsilon-greedy exploration, and
  gradient-clipped Adam optimization.

### train.py

The entry-point script with three modes:

- **`train`** -- Headless training loop. Saves the best model checkpoint
  (by rolling 50-episode average) to `dqn_model.pth` and generates
  `training_plot.png`.
- **`eval`** -- Loads the trained model and renders the agent driving in real
  time at 60 FPS in a Pygame window.
- **`eval-headless`** -- Loads the trained model and runs evaluation episodes
  without graphics, printing per-episode stats and a summary with hit rate,
  average reward, and average steps.

## Installation

```bash
pip install -r requirements.txt
```

For GPU-accelerated training (recommended), install the CUDA build of PyTorch
instead of the CPU-only version:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

## Commands

### Train the agent

```bash
python train.py --mode train
```

Runs 2000 episodes of headless training (default). Progress is printed every
50 episodes. The best model is saved to `dqn_model.pth` and a reward plot is
saved to `training_plot.png`.

Optional flags:

```bash
python train.py --mode train --episodes 3000 --max-steps 250
```

| Flag           | Default | Description                              |
|----------------|---------|------------------------------------------|
| `--episodes`   | 2000    | Number of training episodes              |
| `--max-steps`  | 250     | Maximum steps per episode during training|

### Evaluate with graphics

```bash
python train.py --mode eval
```

Opens a Pygame window and runs 5 episodes at 60 FPS using the trained model
with a fully greedy policy (epsilon = 0). Close the window or wait for all
episodes to finish.

### Evaluate without graphics

```bash
python train.py --mode eval-headless --episodes 100
```

Runs 100 episodes headlessly and prints per-episode results plus a summary:

```
--- Summary ---
Episodes: 100
Targets hit: 85/100 (85.0%)
Avg reward:  72.3
Avg steps:   94
```

### Play manually

```bash
python TestRig.py
```

Drive with WASD. Useful for understanding the task difficulty.


### Approach

The new vision-state feature treats the simulator like a real external system:
instead of reading the car and target positions directly from the game objects,
it periodically captures screenshots of the rendered frame and reconstructs the state from
pixels. The detector uses the known visual appearance of the scene to locate
the playfield, segment the yellow car and blue target, estimate the car's
heading from the blue front marker, and recover the car corners from the car
body geometry.

The first step is locating the playable field inside the screenshot. The
detector searches for pixels close to the dark field color and takes the full
extent of that region as the playfield bounding box. This lets the system work
even if the screenshot includes window borders or title bars, because all later
coordinates are measured relative to the detected field and then translated
back into screenshot coordinates.

The target is detected by thresholding for the blue target color inside that
playfield. After building a boolean mask of matching pixels, the detector keeps
the largest connected blue region and treats that as the target. Its center is
computed from the centroid of all target pixels, which gives the target
coordinates. Its radius is estimated from the pixel area by assuming the blob
is circular.

The car is detected in two parts. First, the yellow body is segmented from the
playfield using the car body color, and the largest connected yellow region is
kept as the car body. The body centroid gives the approximate car center.
Second, the blue front marker is segmented separately using the front-marker
color. Among the blue regions, the detector chooses the one that best matches
the expected marker size and lies closest to the yellow body. The vector from
the yellow body centroid to the blue marker centroid becomes the forward
direction of the car.

Once the forward direction is known, the detector constructs a perpendicular
right-hand axis and projects every yellow-body pixel onto those two axes. The
extreme projections along the forward/backward axis and left/right axis define
the four corners of the oriented car rectangle. Those corner points are then
converted back into screenshot-relative `(x, y)` coordinates by adding the
playfield offset. The heading angle is computed from the forward vector in
screen coordinates, with `0` degrees pointing east and positive rotation
matching the simulator convention.

That screenshot-derived result is then passed back through `Game.getGameState()`
so the rest of the project can keep using the same state interface.

### Automatic Screenshot Capture In `getGameState()`

The screenshot detector is now wired directly into `Game.getGameState()`.

Enable it like this:

```python
from TestRig import Game

game = Game(render_mode="human", use_vision_state=True, capture_interval=1.0 / 15.0)
```

Or through the environment wrapper:

```python
from car_env import CarEnv

env = CarEnv(render_mode="human", use_vision_state=True, capture_interval=1.0 / 15.0)
```

When `use_vision_state=True`:

- the game automatically captures a screenshot every `capture_interval` seconds
- the screenshot is processed by `vision_detector.py`
- `getGameState()` returns the screenshot-derived state instead of the direct geometry state

When `use_vision_state=False`, `getGameState()` keeps using the original exact simulator geometry.

### Recommended Capture Frequency

For a screenshot-driven control loop:

- Recommended: `15-20 Hz` capture + processing
- Acceptable starting point: `10 Hz`
- Usually unnecessary: above `30 Hz` unless you move to a faster vehicle or more aggressive control

Why: the car moves about `220 px/sec`, so at `15 Hz` it advances about `14.7 px`
per decision, which is responsive enough for this map while keeping screenshot
overhead manageable.

For RL specifically, train on simulator state when possible and reserve the
screenshot detector for deployment or imitation-data collection. Training a DQN
from screenshots is possible, but it adds vision noise and slows iteration.

## State Representation

The agent observes a **22-dimensional** normalized float vector extracted from
`getGameState()` each frame:

| Index | Feature                         | Normalization            |
|-------|---------------------------------|--------------------------|
| 0-1   | Front-right corner (x, y)      | x / 900, y / 600        |
| 2-3   | Front-left corner (x, y)       | x / 900, y / 600        |
| 4-5   | Back-right corner (x, y)       | x / 900, y / 600        |
| 6-7   | Back-left corner (x, y)        | x / 900, y / 600        |
| 8     | sin(heading angle)              | Already in [-1, 1]       |
| 9     | cos(heading angle)              | Already in [-1, 1]       |
| 10-11 | Forward direction vector (x, y)| Already unit-length      |
| 12-13 | Target position (x, y)         | x / 900, y / 600        |
| 14    | Target radius                   | radius / 900             |
| 15    | Distance to target              | dist / diagonal          |
| 16-17 | Delta to target (dx, dy)       | dx / diagonal, dy / diag |
| 18    | Distance to left wall           | car_x / 900             |
| 19    | Distance to right wall          | (900 - car_x) / 900     |
| 20    | Distance to top wall            | car_y / 600             |
| 21    | Distance to bottom wall         | (600 - car_y) / 600     |

Features 15-17 (distance and delta to target) are engineered features that give
the network a direct signal about where the target is relative to the car.
Features 18-21 (wall distances) help the agent avoid boundary collisions.

## Reward Function

The reward is computed every step in `CarEnv.step()` using a dense shaping
scheme. Every step begins with a small time penalty, then one of three
outcomes applies:

### Per-step time penalty

```
reward = -0.1
```

Applied unconditionally on every step to encourage the agent to reach the
target quickly rather than wandering.

### Target hit (episode ends successfully)

```
reward += +100.0
```

A large positive bonus when the car's collision mask overlaps the target
circle. The episode terminates and the target is respawned.

### Boundary or obstacle collision (episode ends with failure)

```
reward -= 100.0
```

A large negative penalty when the car drives out of the 900x600 screen bounds
or collides with an obstacle (when obstacles are enabled). The episode
terminates immediately.

### Progress-based shaping (when neither terminal condition is met)

```
progress = (previous_distance - current_distance) / diagonal
reward += progress * 50.0
```

Where `diagonal = sqrt(900^2 + 600^2) = 1082.0`. This gives a small positive
reward each step the car moves closer to the target, and a small negative
reward when it moves further away. The scaling factor of 50.0 keeps progress
rewards meaningful but much smaller than the terminal rewards.

### Reward summary table

| Condition          | Reward Component         | Typical Magnitude |
|--------------------|--------------------------|-------------------|
| Every step         | -0.1                     | -0.1              |
| Hit target         | +100.0                   | +99.9 net         |
| Hit boundary       | -100.0                   | -100.1 net        |
| Moving closer      | +50 * (delta_d / 1082)   | +0.01 to +0.17    |
| Moving away        | +50 * (delta_d / 1082)   | -0.01 to -0.17    |

A successful episode that reaches the target in ~80 steps typically scores
around +95 to +105 total reward. A failed episode that crashes into a wall
scores around -100 to -110.

## DQN Training Details

### Algorithm: Double DQN

Uses two networks -- a **policy network** that is trained, and a **target
network** that provides stable Q-value targets. The Double DQN variant uses
the policy network to *select* the best next action and the target network to
*evaluate* it, eliminating the overestimation bias of vanilla DQN.

### Hyperparameters

| Parameter          | Value    | Description                                    |
|--------------------|----------|------------------------------------------------|
| Learning rate      | 5e-4     | Adam optimizer step size                       |
| Discount (gamma)   | 0.99     | Future reward discount factor                  |
| Batch size         | 128      | Transitions sampled per learning step          |
| Buffer capacity    | 200,000  | Maximum replay buffer size                     |
| Epsilon start      | 1.0      | Initial exploration rate (100% random)         |
| Epsilon end        | 0.01     | Minimum exploration rate                       |
| Epsilon decay      | 0.997    | Multiplicative decay per episode               |
| Tau                | 0.005    | Soft target update rate (Polyak averaging)     |
| Learn every        | 4 steps  | NN update frequency (skips reduce compute)     |
| Gradient clip      | 10.0     | Max gradient norm to prevent exploding updates |
| Loss function      | Huber    | SmoothL1Loss, robust to outlier rewards        |

### Training flow

1. The agent starts with epsilon = 1.0 (fully random actions).
2. Each episode: `reset()` the environment, then loop `select_action` ->
   `step` -> `store_transition` -> `learn` (every 4 steps) until the episode
   ends.
3. After each episode, epsilon is decayed by multiplying by 0.997.
4. Every 50 episodes, the rolling 50-episode average reward is checked. If it
   is the best seen so far, the model weights are saved to `dqn_model.pth`.
5. After all episodes, a matplotlib plot of episode rewards (with rolling
   average) is saved to `training_plot.png`.

## Enabling Obstacles

Obstacle support is built in but disabled. To enable it, change line 45 of
`TestRig.py`:

```python
Obstacles = True
```

The environment and reward function handle obstacles automatically -- obstacle
collisions trigger the same termination and -100 penalty as boundary
collisions. No other code changes are needed. The agent will need to be
retrained after enabling obstacles.

## Dependencies

- Python 3.10+
- pygame >= 2.5.0
- torch >= 2.0.0 (CUDA build recommended for GPU training)
- numpy >= 1.24.0
- matplotlib >= 3.7.0
