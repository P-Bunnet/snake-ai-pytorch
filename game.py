import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Enum for direction
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
PURPLE = (255, 0, 255)

BLOCK_SIZE = 20
SPEED = 20
GROWTH_RATE = 2  # Number of segments added per food

class SnakeGameAI:
    def __init__(self, w=640, h=480, num_walls=3, num_poison=2):
        self.w = w
        self.h = h
        self.num_walls = num_walls
        self.num_poison = num_poison
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI with Obstacles')
        self.clock = pygame.time.Clock()

        # ✅ Load images (Ensure they are in the `assets` folder)
        self.apple_img = pygame.image.load("assets/apple.png")
        self.poison_img = pygame.image.load("assets/poison.png")
        self.wall_img = pygame.image.load("assets/wall.jpg")

        # ✅ Scale images to fit BLOCK_SIZE
        self.apple_img = pygame.transform.scale(self.apple_img, (BLOCK_SIZE, BLOCK_SIZE))
        self.poison_img = pygame.transform.scale(self.poison_img, (BLOCK_SIZE, BLOCK_SIZE))
        self.wall_img = pygame.transform.scale(self.wall_img, (BLOCK_SIZE, BLOCK_SIZE))

        self.reset()


    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self.walls = []
        self.poison = []
        self._place_food()
        self._place_walls()
        self._place_poison()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            food = Point(x, y)
            if food not in self.snake and food not in self.walls and food not in self.poison:
                self.food = food
                break

    def _place_walls(self):
        self.walls = []
        for _ in range(self.num_walls):
            while True:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                wall = Point(x, y)
                if wall not in self.snake and wall != self.food:
                    self.walls.append(wall)
                    break

    def _place_poison(self):
        self.poison = []
        for _ in range(self.num_poison):
            while True:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                poison = Point(x, y)
                if poison not in self.snake and poison != self.food and poison not in self.walls:
                    self.poison.append(poison)
                    break

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        
        self_collide, wall_border_collide, wall_placed_collide = self._is_collision()
        
        if wall_border_collide or self.frame_iteration > 150 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        if wall_placed_collide:
            game_over = True
            reward = -5
            return reward, game_over, self.score
        
        if self_collide:
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        if self.head == self.food:
            self.score += 1
            reward = 15
            for _ in range(GROWTH_RATE):  
                self.snake.append(self.snake[-1])
            self._place_food()
        else:
            self.snake.pop()

        if self.head in self.poison:
            reward = -10
            self.score -= 0.5
            self.snake = self.snake[:-2] if len(self.snake) > 3 else self.snake
            self.poison.remove(self.head)

        # Try to avoid the poison
        min_distance = float('inf')
        for p in self.poison:
            distance = abs(self.head.x - p.x) + abs(self.head.y - p.y)
            min_distance = min(min_distance, distance)
        
        if min_distance < BLOCK_SIZE:
            reward = - 1 # stronger penalty for being next to poison
        if min_distance <= 2 * BLOCK_SIZE:
            reward = - 0.5 # medium penalty for getting close to poison
        if min_distance <= 3 * BLOCK_SIZE:
            reward = - 0.1 # small penalty for being somewhat close to poison
            
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # ✅ Detect border collision
        wall_border_collide = pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0

        # ✅ Detect self-collision
        self_collide = pt in self.snake[1:]

        # ✅ Detect placed wall collision (NEW)
        wall_placed_collide = pt in self.walls


        return self_collide, wall_border_collide, wall_placed_collide



    def _update_ui(self):
        self.display.fill(BLACK)

        # ✅ Draw Snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # ✅ Draw Apple (Food)
        self.display.blit(self.apple_img, (self.food.x, self.food.y))

        # ✅ Draw Walls 🧱
        for wall in self.walls:
            self.display.blit(self.wall_img, (wall.x, wall.y))

        # ✅ Draw Poison ☠️
        for poison in self.poison:
            self.display.blit(self.poison_img, (poison.x, poison.y))

        # ✅ Draw Score
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])

        pygame.display.flip()


    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir
        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
