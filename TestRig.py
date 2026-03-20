"""
Simple 2D Top-Down Driving (Pygame)
pip install pygame
python main.py
"""

import math
import random
import pygame


# ----------------------------
# Constants
# ----------------------------
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
FPS = 60

FIELD_COLOR = (40, 45, 50)
BORDER_COLOR = (80, 85, 90)
CAR_COLOR = (255, 220, 40)
CAR_FRONT_COLOR = (30, 200, 255)
TARGET_COLOR = (60, 220, 255)
OBSTACLE_COLOR = (220, 70, 70)

CAR_LENGTH = 48
CAR_WIDTH = 26
CAR_SPEED = 220.0          # pixels/sec
CAR_ROT_SPEED = 160.0      # degrees/sec

TARGET_RADIUS = 16

MIN_OBSTACLES = 8
MAX_OBSTACLES = 12

SPAWN_MARGIN = 24
MIN_CAR_TARGET_DIST = 180
MIN_CAR_OBS_DIST = 100
MIN_TARGET_OBS_DIST = 80
OBSTACLE_PADDING = 8

MAX_PLACEMENT_ATTEMPTS = 200

# Toggle obstacle spawning on/off.
Obstacles = False


# ----------------------------
# Utility
# ----------------------------
def clamp(value, lo, hi):
    return max(lo, min(value, hi))


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def random_position(margin):
    x = random.uniform(margin, SCREEN_WIDTH - margin)
    y = random.uniform(margin, SCREEN_HEIGHT - margin)
    return (x, y)


# ----------------------------
# Entities
# ----------------------------
class Car:
    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        # Heading angle in degrees, 0 points to the right, positive rotates CCW
        self.angle = 0.0
        self.speed = CAR_SPEED
        self.rotation_speed = CAR_ROT_SPEED

        self.base_surface = pygame.Surface((CAR_LENGTH, CAR_WIDTH), pygame.SRCALPHA)
        pygame.draw.rect(self.base_surface, CAR_COLOR, self.base_surface.get_rect(), border_radius=4)
        # Front marker (triangle) on the right edge to indicate heading
        mid_y = CAR_WIDTH / 2
        nose_tip = (CAR_LENGTH - 2, mid_y)
        nose_left = (CAR_LENGTH - 12, mid_y - 6)
        nose_right = (CAR_LENGTH - 12, mid_y + 6)
        pygame.draw.polygon(self.base_surface, CAR_FRONT_COLOR, [nose_tip, nose_left, nose_right])

        # Previous state for collision rollback
        self.prev_pos = self.pos.copy()
        self.prev_angle = self.angle

    def forward_vector(self):
        # +y is down in screen space, so use -sin for CCW-positive heading
        rad = math.radians(self.angle)
        return pygame.Vector2(math.cos(rad), -math.sin(rad))

    def update(self, dt, action):
        # Store previous state
        self.prev_pos = self.pos.copy()
        self.prev_angle = self.angle

        if action == 2:
            self.angle += self.rotation_speed * dt
        if action == 3:
            self.angle -= self.rotation_speed * dt

        direction = self.forward_vector()
        if action == 0:
            self.pos += direction * self.speed * dt
        if action == 1:
            self.pos -= direction * self.speed * dt

    def get_render_data(self):
        rotated = pygame.transform.rotate(self.base_surface, self.angle)
        rect = rotated.get_rect(center=(self.pos.x, self.pos.y))
        mask = pygame.mask.from_surface(rotated)
        return rotated, rect, mask

    def revert(self):
        self.pos = self.prev_pos
        self.angle = self.prev_angle


class Target:
    def __init__(self, pos, radius):
        self.radius = radius
        self.surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.surface, TARGET_COLOR, (radius, radius), radius)
        self.mask = pygame.mask.from_surface(self.surface)
        self.rect = self.surface.get_rect(center=pos)

    def set_pos(self, pos):
        self.rect.center = pos

    def draw(self, screen):
        screen.blit(self.surface, self.rect)


class Obstacle:
    def __init__(self, shape_type, center, size_params, color):
        self.shape_type = shape_type
        self.color = color
        self.surface, self.mask, self.radius = self._create_surface(shape_type, size_params)
        self.rect = self.surface.get_rect(center=center)

    def _create_surface(self, shape_type, size_params):
        if shape_type == "rect":
            w, h = size_params
            surface = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(surface, self.color, surface.get_rect(), border_radius=3)
            radius = 0.5 * max(w, h)
            return surface, pygame.mask.from_surface(surface), radius

        if shape_type == "circle":
            r = size_params
            surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, self.color, (r, r), r)
            return surface, pygame.mask.from_surface(surface), r

        if shape_type == "triangle":
            w, h = size_params
            surface = pygame.Surface((w, h), pygame.SRCALPHA)
            points = [(w / 2, 0), (0, h), (w, h)]
            pygame.draw.polygon(surface, self.color, points)
            radius = 0.5 * max(w, h)
            return surface, pygame.mask.from_surface(surface), radius

        if shape_type == "poly":
            w, h = size_params
            surface = pygame.Surface((w, h), pygame.SRCALPHA)
            cx, cy = w / 2, h / 2
            points = []
            count = random.randint(5, 7)
            angles = sorted(random.uniform(0, 2 * math.pi) for _ in range(count))
            max_r = min(w, h) * 0.5
            for ang in angles:
                r = random.uniform(max_r * 0.7, max_r)
                x = cx + math.cos(ang) * r
                y = cy + math.sin(ang) * r
                points.append((x, y))
            pygame.draw.polygon(surface, self.color, points)
            radius = 0.5 * max(w, h)
            return surface, pygame.mask.from_surface(surface), radius

        # Fallback
        surface = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.rect(surface, self.color, surface.get_rect())
        return surface, pygame.mask.from_surface(surface), 20

    def draw(self, screen):
        screen.blit(self.surface, self.rect)


# ----------------------------
# Game
# ----------------------------
class Game:
    def __init__(self, render_mode="human"):
        pygame.init()
        self.render_mode = render_mode
        if render_mode == "human":
            pygame.display.set_caption("Top-Down Driving")
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.font = pygame.font.Font(None, 24)
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.font = None
        self.dt = 1.0 / FPS

        self.car_spawn = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        self.car = Car(self.car_spawn)
        self.target = Target((0, 0), TARGET_RADIUS)
        self.obstacles = []
        self.score = 0

        self.reset(collected=False)

    def reset(self, collected):
        if collected:
            self.score += 1

        self.car.pos = pygame.Vector2(self.car_spawn)
        self.car.angle = 0.0

        self.obstacles = self.generate_obstacles()
        self.place_target()
        return self.getGameState()

    def generate_obstacles(self):
        if not Obstacles:
            return []

        count = random.randint(MIN_OBSTACLES, MAX_OBSTACLES)
        shape_types = ["rect", "circle", "triangle", "poly"]

        # Ensure at least 3 distinct types appear
        base_types = random.sample(shape_types, 3)
        while len(base_types) < count:
            base_types.append(random.choice(shape_types))
        random.shuffle(base_types)

        for attempt_round in range(6):
            padding = OBSTACLE_PADDING * (0.8 ** attempt_round)
            obstacles = []
            success = True

            for i in range(count):
                shape = base_types[i]
                placed = False

                for _ in range(MAX_PLACEMENT_ATTEMPTS):
                    if shape == "rect":
                        w = random.randint(40, 90)
                        h = random.randint(30, 80)
                        size = (w, h)
                        radius = 0.5 * max(w, h)
                    elif shape == "circle":
                        r = random.randint(18, 40)
                        size = r
                        radius = r
                    elif shape == "triangle":
                        w = random.randint(40, 90)
                        h = random.randint(40, 90)
                        size = (w, h)
                        radius = 0.5 * max(w, h)
                    else:  # poly
                        w = random.randint(50, 100)
                        h = random.randint(50, 100)
                        size = (w, h)
                        radius = 0.5 * max(w, h)

                    pos = random_position(radius + SPAWN_MARGIN)

                    if distance(pos, self.car_spawn) < (radius + MIN_CAR_OBS_DIST):
                        continue

                    overlap = False
                    for o in obstacles:
                        if distance(pos, o.rect.center) < (radius + o.radius + padding):
                            overlap = True
                            break
                    if overlap:
                        continue

                    obstacles.append(Obstacle(shape, pos, size, OBSTACLE_COLOR))
                    placed = True
                    break

                if not placed:
                    success = False
                    break

            if success:
                return obstacles

        # Fallback: return whatever was placed
        return obstacles

    def place_target(self):
        for attempt_round in range(4):
            padding = MIN_TARGET_OBS_DIST * (0.8 ** attempt_round)

            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                pos = random_position(TARGET_RADIUS + SPAWN_MARGIN)

                if distance(pos, self.car_spawn) < MIN_CAR_TARGET_DIST:
                    continue

                bad = False
                for o in self.obstacles:
                    if distance(pos, o.rect.center) < (TARGET_RADIUS + o.radius + padding):
                        bad = True
                        break
                if bad:
                    continue

                self.target.set_pos(pos)
                return

        # Fallback: place without obstacle checks
        self.target.set_pos(random_position(TARGET_RADIUS + SPAWN_MARGIN))

    def car_hits_obstacle(self, car_mask, car_rect):
        for obs in self.obstacles:
            if not car_rect.colliderect(obs.rect):
                continue
            offset = (obs.rect.left - car_rect.left, obs.rect.top - car_rect.top)
            if car_mask.overlap(obs.mask, offset):
                return True
        return False

    def car_hits_target(self, car_mask, car_rect):
        if not car_rect.colliderect(self.target.rect):
            return False
        offset = (self.target.rect.left - car_rect.left, self.target.rect.top - car_rect.top)
        return car_mask.overlap(self.target.mask, offset) is not None

    def keep_car_in_bounds(self, car_rect):
        if car_rect.left < 0 or car_rect.right > SCREEN_WIDTH:
            return False
        if car_rect.top < 0 or car_rect.bottom > SCREEN_HEIGHT:
            return False
        return True

    def draw_ui(self):
        if self.render_mode != "human":
            return
        text = "WASD to drive, Q to quit"
        score_text = f"Targets: {self.score}"
        surf = self.font.render(text, True, (230, 230, 230))
        score_surf = self.font.render(score_text, True, (230, 230, 230))
        self.screen.blit(surf, (10, 10))
        self.screen.blit(score_surf, (10, 32))

    def getGameState(self):
        center = self.car.pos
        half_length = CAR_LENGTH / 2.0
        half_width = CAR_WIDTH / 2.0

        forward = self.car.forward_vector()
        right = pygame.Vector2(forward.y, -forward.x)

        front_center = center + forward * half_length
        back_center = center - forward * half_length

        front_right = front_center + right * half_width
        front_left = front_center - right * half_width
        back_right = back_center + right * half_width
        back_left = back_center - right * half_width

        return {
            "car_corners": {
                "front_right": (front_right.x, front_right.y),
                "back_left": (back_left.x, back_left.y),
                "back_right": (back_right.x, back_right.y),
                "front_left": (front_left.x, front_left.y),
            },
            "car_direction": {
                "angle_degrees": self.car.angle,
                "forward_vector": (forward.x, forward.y),
            },
            "target": {
                "coordinates": self.target.rect.center,
                "radius": self.target.radius,
            },
        }

    def step(self, action):
        pygame.event.pump()
        self.car.update(self.dt, action)

        terminated = False
        hit_target = False

        # Collision checks
        _car_surface, car_rect, car_mask = self.car.get_render_data()
        hit_obstacle = self.car_hits_obstacle(car_mask, car_rect)
        out_of_bounds = not self.keep_car_in_bounds(car_rect)
        hit_target = self.car_hits_target(car_mask, car_rect)

        if hit_obstacle or out_of_bounds or hit_target:
            terminated = True

        if hit_obstacle or out_of_bounds:
            self.car.revert()

        if hit_target:
            self.reset(collected=True)

        return self.getGameState(), terminated, hit_target

    def render(self):
        if self.render_mode != "human":
            return
        car_surface, car_rect, _car_mask = self.car.get_render_data()
        self.screen.fill(FIELD_COLOR)
        pygame.draw.rect(self.screen, BORDER_COLOR, self.screen.get_rect(), 2)

        for obs in self.obstacles:
            obs.draw(self.screen)

        self.target.draw(self.screen)
        self.screen.blit(car_surface, car_rect)
        self.draw_ui()
        pygame.display.flip()


if __name__ == "__main__":
    game = Game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        action = 4
        if keys[pygame.K_w]:
            action = 0
        elif keys[pygame.K_s]:
            action = 1
        elif keys[pygame.K_a]:
            action = 2
        elif keys[pygame.K_d]:
            action = 3
        game.step(action)
        game.render()

    pygame.quit()
