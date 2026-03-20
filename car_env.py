"""
Gym-style wrapper around the TestRig Game class for RL training.
"""

import math
import numpy as np
from TestRig import Game, SCREEN_WIDTH, SCREEN_HEIGHT

DIAGONAL = math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)
STATE_SIZE = 22
NUM_ACTIONS = 5


class CarEnv:
    """OpenAI-Gym-compatible environment wrapping the top-down driving game."""

    def __init__(self, render_mode=None, max_steps=1000):
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.game = Game(render_mode=render_mode or "headless")
        self.steps = 0
        self.prev_distance = None

    def _flatten_state(self, game_state):
        corners = game_state["car_corners"]
        direction = game_state["car_direction"]
        target = game_state["target"]

        fr = corners["front_right"]
        fl = corners["front_left"]
        br = corners["back_right"]
        bl = corners["back_left"]

        car_center_x = (fr[0] + fl[0] + br[0] + bl[0]) / 4.0
        car_center_y = (fr[1] + fl[1] + br[1] + bl[1]) / 4.0
        tx, ty = target["coordinates"]

        dx = tx - car_center_x
        dy = ty - car_center_y
        dist = math.hypot(dx, dy)

        angle_rad = math.radians(direction["angle_degrees"])

        wall_left = car_center_x / SCREEN_WIDTH
        wall_right = (SCREEN_WIDTH - car_center_x) / SCREEN_WIDTH
        wall_top = car_center_y / SCREEN_HEIGHT
        wall_bottom = (SCREEN_HEIGHT - car_center_y) / SCREEN_HEIGHT

        state = np.array([
            fr[0] / SCREEN_WIDTH,
            fr[1] / SCREEN_HEIGHT,
            fl[0] / SCREEN_WIDTH,
            fl[1] / SCREEN_HEIGHT,
            br[0] / SCREEN_WIDTH,
            br[1] / SCREEN_HEIGHT,
            bl[0] / SCREEN_WIDTH,
            bl[1] / SCREEN_HEIGHT,
            math.sin(angle_rad),
            math.cos(angle_rad),
            direction["forward_vector"][0],
            direction["forward_vector"][1],
            tx / SCREEN_WIDTH,
            ty / SCREEN_HEIGHT,
            target["radius"] / max(SCREEN_WIDTH, SCREEN_HEIGHT),
            dist / DIAGONAL,
            dx / DIAGONAL,
            dy / DIAGONAL,
            wall_left,
            wall_right,
            wall_top,
            wall_bottom,
        ], dtype=np.float32)

        return state, dist

    def reset(self):
        self.game.score = 0
        game_state = self.game.reset(collected=False)
        self.steps = 0
        state, dist = self._flatten_state(game_state)
        self.prev_distance = dist
        return state

    def step(self, action):
        game_state, terminated, hit_target = self.game.step(action)
        self.steps += 1

        state, dist = self._flatten_state(game_state)

        reward = -0.1

        if hit_target:
            reward += 100.0
        elif terminated:
            reward -= 100.0
        else:
            progress = (self.prev_distance - dist) / DIAGONAL
            reward += progress * 50.0

        self.prev_distance = dist

        truncated = self.steps >= self.max_steps
        done = terminated or truncated

        info = {
            "hit_target": hit_target,
            "terminated": terminated,
            "truncated": truncated,
            "steps": self.steps,
        }

        return state, reward, done, truncated, info

    def render(self):
        self.game.render()

    def close(self):
        import pygame
        pygame.quit()
