#
# def dfs(current_position, target, maze):
#     stack = Stack()
#     visited = Set()
#     parent = Dictionary()  # To track path
#
#     stack.push(current_position)
#     visited.add(current_position)
#
#     while not stack.is_empty():
#         current = stack.pop()
#
#         # If we found the target
#         if current == target:
#             return reconstruct_path(parent, current_position, target)
#
#         # Explore neighbors (usually in fixed order: UP, RIGHT, DOWN, LEFT)
#         for neighbor in get_neighbors(current, maze):
#             if neighbor not in visited and not is_wall(neighbor):
#                 visited.add(neighbor)
#                 parent[neighbor] = current
#                 stack.push(neighbor)
#
#     return []  # No path found

##########################################

import pygame
import sys
import time
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
CELL_SIZE = 30
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE
FPS = 10

# Colors
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class Maze:
    def __init__(self):
        self.walls = set()
        self.food = set()
        self.pacman_pos = (1, 1)
        self.ghost_pos = (ROWS - 2, COLS - 2)
        self.score = 0
        self.game_over = False
        self.create_maze()

    def create_maze(self):
        # Create border walls
        for i in range(ROWS):
            self.walls.add((i, 0))
            self.walls.add((i, COLS - 1))
        for j in range(COLS):
            self.walls.add((0, j))
            self.walls.add((ROWS - 1, j))

        # Create some internal walls
        internal_walls = [
            # Vertical walls
            [(2, j) for j in range(2, 8)],
            [(5, j) for j in range(3, 10)],
            [(8, j) for j in range(5, 15)],
            [(12, j) for j in range(2, 12)],
            # Horizontal walls
            [(i, 4) for i in range(3, 8)],
            [(i, 8) for i in range(6, 12)],
            [(i, 12) for i in range(4, 10)],
            [(i, 15) for i in range(8, 14)]
        ]

        for wall_segment in internal_walls:
            self.walls.update(wall_segment)

        # Add food pellets (all empty spaces that aren't walls or pacman/ghost start)
        for i in range(1, ROWS - 1):
            for j in range(1, COLS - 1):
                pos = (i, j)
                if (pos not in self.walls and
                        pos != self.pacman_pos and
                        pos != self.ghost_pos):
                    self.food.add(pos)

    def is_wall(self, position):
        return position in self.walls

    def is_valid_position(self, position):
        i, j = position
        return 0 <= i < ROWS and 0 <= j < COLS

    def has_food(self, position):
        return position in self.food

    def remove_food(self, position):
        if position in self.food:
            self.food.remove(position)
            self.score += 10
            return True
        return False

    def get_food_positions(self):
        return list(self.food)

    def get_pacman_position(self):
        return self.pacman_pos

    def move_pacman(self, direction):
        if self.game_over:
            return

        i, j = self.pacman_pos
        new_pos = self.pacman_pos

        if direction == 'NORTH':
            new_pos = (i - 1, j)
        elif direction == 'SOUTH':
            new_pos = (i + 1, j)
        elif direction == 'EAST':
            new_pos = (i, j + 1)
        elif direction == 'WEST':
            new_pos = (i, j - 1)

        if self.is_valid_position(new_pos) and not self.is_wall(new_pos):
            self.pacman_pos = new_pos

            # Check for food
            if self.has_food(new_pos):
                self.remove_food(new_pos)

            # Check win condition
            if len(self.food) == 0:
                self.game_over = True
                print("You Win! Final Score:", self.score)

            # Check ghost collision
            if self.pacman_pos == self.ghost_pos:
                self.game_over = True
                print("Game Over! Final Score:", self.score)


class DFSPacman:
    def __init__(self, maze):
        self.maze = maze
        self.path = []
        self.visited_nodes = set()
        self.exploration_path = []

    def dfs_search(self, start, target):
        """Perform DFS from start to target"""
        stack = [start]
        visited = {start}
        parent = {start: None}

        while stack:
            current = stack.pop()
            self.visited_nodes.add(current)  # For visualization

            # Found the target
            if current == target:
                return self.reconstruct_path(parent, start, target)

            # Get neighbors in consistent order
            neighbors = self.get_neighbors(current)
            # Reverse to maintain consistent exploration order
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    stack.append(neighbor)

        return []  # No path found

    def get_neighbors(self, position):
        """Get valid neighboring positions in fixed order"""
        i, j = position
        neighbors = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W

        for di, dj in directions:
            new_i, new_j = i + di, j + dj
            new_pos = (new_i, new_j)

            if (self.maze.is_valid_position(new_pos) and
                    not self.maze.is_wall(new_pos)):
                neighbors.append(new_pos)

        return neighbors

    def reconstruct_path(self, parent, start, target):
        """Build path from start to target using parent pointers"""
        path = []
        current = target

        while current != start:
            path.append(current)
            current = parent[current]

        path.reverse()  # Reverse to get start -> target order
        return path

    def find_path_to_food(self, pacman_position):
        """Find path to nearest food using DFS"""
        food_positions = self.maze.get_food_positions()

        if not food_positions:
            return []

        # Try to find path to each food until we find a reachable one
        for food in food_positions:
            path = self.dfs_search(pacman_position, food)
            if path:
                return path

        return []

    def get_next_move(self, pacman_position):
        """Get the next move for PACMAN"""
        # If we don't have a path or current path is invalid, find new path
        if not self.path or pacman_position not in [self.path[0]] + [pacman_position]:
            self.path = self.find_path_to_food(pacman_position)
            self.visited_nodes.clear()  # Reset for new search

        if self.path:
            next_position = self.path[0]
            self.path = self.path[1:]  # Remove the first element
            return self.get_direction(pacman_position, next_position)

        return 'STOP'  # No move available

    def get_direction(self, current, next_pos):
        """Convert position difference to direction"""
        curr_i, curr_j = current
        next_i, next_j = next_pos

        if next_j > curr_j: return 'EAST'
        if next_j < curr_j: return 'WEST'
        if next_i > curr_i: return 'SOUTH'
        if next_i < curr_i: return 'NORTH'

        return 'STOP'


class Ghost:
    def __init__(self, maze):
        self.maze = maze
        self.position = maze.ghost_pos
        self.directions = ['NORTH', 'SOUTH', 'EAST', 'WEST']

    def move_random(self):
        """Simple random movement for ghost"""
        i, j = self.position
        possible_moves = []

        for direction in self.directions:
            if direction == 'NORTH':
                new_pos = (i - 1, j)
            elif direction == 'SOUTH':
                new_pos = (i + 1, j)
            elif direction == 'EAST':
                new_pos = (i, j + 1)
            elif direction == 'WEST':
                new_pos = (i, j - 1)

            if (self.maze.is_valid_position(new_pos) and
                    not self.maze.is_wall(new_pos)):
                possible_moves.append((direction, new_pos))

        if possible_moves:
            import random
            direction, new_pos = random.choice(possible_moves)
            self.position = new_pos

            # Check if ghost caught pacman
            if self.position == self.maze.pacman_pos:
                self.maze.game_over = True
                print("Ghost caught you! Game Over!")


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("PACMAN with DFS")
        self.clock = pygame.time.Clock()
        self.maze = Maze()
        self.dfs_pacman = DFSPacman(self.maze)
        self.ghost = Ghost(self.maze)
        self.font = pygame.font.Font(None, 36)

    def draw_maze(self):
        self.screen.fill(BLACK)

        # Draw walls
        for wall in self.maze.walls:
            i, j = wall
            pygame.draw.rect(self.screen, BLUE,
                             (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Draw food
        for food in self.maze.food:
            i, j = food
            pygame.draw.circle(self.screen, WHITE,
                               (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2),
                               CELL_SIZE // 8)

        # Draw visited nodes (DFS exploration)
        for node in self.dfs_pacman.visited_nodes:
            i, j = node
            if (i, j) != self.maze.pacman_pos and (i, j) not in self.maze.food:
                pygame.draw.rect(self.screen, (50, 50, 50),
                                 (j * CELL_SIZE + CELL_SIZE // 4, i * CELL_SIZE + CELL_SIZE // 4,
                                  CELL_SIZE // 2, CELL_SIZE // 2))

        # Draw path
        if hasattr(self.dfs_pacman, 'path'):
            for pos in self.dfs_pacman.path:
                i, j = pos
                pygame.draw.rect(self.screen, GREEN,
                                 (j * CELL_SIZE + CELL_SIZE // 3, i * CELL_SIZE + CELL_SIZE // 3,
                                  CELL_SIZE // 3, CELL_SIZE // 3))

        # Draw pacman
        i, j = self.maze.pacman_pos
        pygame.draw.circle(self.screen, YELLOW,
                           (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2),
                           CELL_SIZE // 2 - 2)

        # Draw ghost
        i, j = self.ghost.position
        pygame.draw.circle(self.screen, RED,
                           (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2),
                           CELL_SIZE // 2 - 2)

        # Draw score
        score_text = self.font.render(f"Score: {self.maze.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # Draw game over message
        if self.maze.game_over:
            game_over_text = self.font.render("GAME OVER", True, WHITE)
            self.screen.blit(game_over_text, (WIDTH // 2 - 80, HEIGHT // 2))

    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game
                        self.maze = Maze()
                        self.dfs_pacman = DFSPacman(self.maze)
                        self.ghost = Ghost(self.maze)

            if not self.maze.game_over:
                # Get DFS move for pacman
                pacman_pos = self.maze.get_pacman_position()
                move = self.dfs_pacman.get_next_move(pacman_pos)
                self.maze.move_pacman(move)

                # Move ghost (every 2 frames to make it slower)
                if pygame.time.get_ticks() % 2 == 0:
                    self.ghost.move_random()

            self.draw_maze()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()