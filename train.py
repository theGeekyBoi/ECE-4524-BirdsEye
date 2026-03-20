"""
Training and evaluation entry-point for the DQN driving agent.

Usage:
    python train.py --mode train          # headless training (default)
    python train.py --mode eval           # visualise the trained agent
    python train.py --mode train --episodes 5000
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygame

from car_env import CarEnv
from dqn_agent import DQNAgent

MODEL_PATH = "dqn_model.pth"
PLOT_PATH = "training_plot.png"


def rolling_average(data, window=50):
    if len(data) < window:
        window = max(1, len(data))
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    out = np.empty_like(data, dtype=float)
    out[:window] = cumsum[:window] / np.arange(1, window + 1)
    out[window:] = cumsum[window:] / window
    return out


def plot_scores(scores, path=PLOT_PATH):
    episodes = np.arange(1, len(scores) + 1)
    scores_arr = np.array(scores, dtype=float)
    avg = rolling_average(scores_arr, window=50)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, scores_arr, alpha=0.3, label="Episode reward")
    ax.plot(episodes, avg, linewidth=2, label="Rolling avg (50)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN Training — Score vs Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}")


LEARN_EVERY = 4


def train(num_episodes=2000, max_steps=250):
    env = CarEnv(render_mode=None, max_steps=max_steps)
    agent = DQNAgent()

    scores = []
    best_avg = -float("inf")
    total_steps = 0

    print(f"Training for {num_episodes} episodes on {agent.device} ...")
    t_start = time.time()

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, float(done))
            total_steps += 1
            if total_steps % LEARN_EVERY == 0:
                agent.learn()
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        scores.append(total_reward)

        if ep % 50 == 0:
            recent = np.mean(scores[-50:])
            elapsed = time.time() - t_start
            print(
                f"Ep {ep:>5d} | reward {total_reward:>8.1f} | "
                f"avg50 {recent:>8.1f} | eps {agent.epsilon:.3f} | "
                f"buf {len(agent.replay_buffer):>6d} | {elapsed:.0f}s"
            )
            if recent > best_avg:
                best_avg = recent
                agent.save(MODEL_PATH)

    env.close()
    print(f"Training complete. Best model saved to {MODEL_PATH}")
    plot_scores(scores)


def evaluate(num_episodes=5, max_steps=500):
    env = CarEnv(render_mode="human", max_steps=max_steps)
    agent = DQNAgent()
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0

    clock = pygame.time.Clock()

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action = agent.select_action(state, evaluate=True)
            state, reward, done, _truncated, info = env.step(action)
            total_reward += reward
            env.render()
            clock.tick(60)

        print(
            f"Eval episode {ep} | reward {total_reward:.1f} | "
            f"target {'HIT' if info.get('hit_target') else 'MISSED'} | "
            f"steps {info.get('steps')}"
        )
        pygame.time.wait(500)

    env.close()


def evaluate_headless(num_episodes=20, max_steps=500):
    """Run the trained agent without any graphics — prints stats only."""
    env = CarEnv(render_mode=None, max_steps=max_steps)
    agent = DQNAgent()
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0

    hits = 0
    total_steps = 0
    total_reward_sum = 0.0

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, _truncated, info = env.step(action)
            total_reward += reward

        hit = info.get("hit_target", False)
        steps = info.get("steps", 0)
        hits += int(hit)
        total_steps += steps
        total_reward_sum += total_reward

        print(
            f"Episode {ep:>3d} | reward {total_reward:>8.1f} | "
            f"steps {steps:>4d} | target {'HIT' if hit else 'MISSED'}"
        )

    print("\n--- Summary ---")
    print(f"Episodes: {num_episodes}")
    print(f"Targets hit: {hits}/{num_episodes} ({100 * hits / num_episodes:.1f}%)")
    print(f"Avg reward:  {total_reward_sum / num_episodes:.1f}")
    print(f"Avg steps:   {total_steps / num_episodes:.0f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Driving Agent")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "eval-headless"],
        default="train",
        help="'train' for headless training, 'eval' to visualise, "
             "'eval-headless' to test without graphics",
    )
    parser.add_argument(
        "--episodes", type=int, default=2000, help="Number of episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=250, help="Max steps per episode (training)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(num_episodes=args.episodes, max_steps=args.max_steps)
    elif args.mode == "eval":
        evaluate()
    else:
        evaluate_headless(num_episodes=args.episodes)
