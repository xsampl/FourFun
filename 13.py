import pygame
import random
import numpy as np
import torch
from collections import deque

import torch.nn as nn
import torch.optim as optim

# --- Snake Game ---
WIDTH, HEIGHT = 400, 400
BLOCK_SIZE = 20
FPS = 60

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(WIDTH//2, HEIGHT//2)]
        self.direction = random.choice([(0, -BLOCK_SIZE), (0, BLOCK_SIZE), (-BLOCK_SIZE, 0), (BLOCK_SIZE, 0)])
        self.spawn_food()
        self.score = 0
        self.done = False

    def spawn_food(self):
        while True:
            x = random.randrange(0, WIDTH, BLOCK_SIZE)
            y = random.randrange(0, HEIGHT, BLOCK_SIZE)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def step(self, action):
        # action: 0=straight, 1=right, 2=left
        dx, dy = self.direction
        if action == 1:  # right turn
            dx, dy = dy, -dx
        elif action == 2:  # left turn
            dx, dy = -dy, dx
        self.direction = (dx, dy)
        head = (self.snake[0][0] + dx, self.snake[0][1] + dy)

        # Check collision
        if (head[0] < 0 or head[0] >= WIDTH or head[1] < 0 or head[1] >= HEIGHT or head in self.snake):
            self.done = True
            reward = -10
            return self.get_state(), reward, self.done

        self.snake.insert(0, head)
        if head == self.food:
            self.score += 1
            reward = 10
            self.spawn_food()
        else:
            self.snake.pop()
            reward = 0
        return self.get_state(), reward, self.done

    def get_state(self):
        head = self.snake[0]
        point_l = (head[0] - self.direction[1], head[1] + self.direction[0])
        point_r = (head[0] + self.direction[1], head[1] - self.direction[0])
        point_s = (head[0] + self.direction[0], head[1] + self.direction[1])

        danger_straight = (point_s in self.snake or
                           point_s[0] < 0 or point_s[0] >= WIDTH or
                           point_s[1] < 0 or point_s[1] >= HEIGHT)
        danger_right = (point_r in self.snake or
                        point_r[0] < 0 or point_r[0] >= WIDTH or
                        point_r[1] < 0 or point_r[1] >= HEIGHT)
        danger_left = (point_l in self.snake or
                       point_l[0] < 0 or point_l[0] >= WIDTH or
                       point_l[1] < 0 or point_l[1] >= HEIGHT)

        food_dir = [
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1],  # food down
        ]
        state = [
            danger_straight, danger_right, danger_left,
            self.direction == (0, -BLOCK_SIZE),  # up
            self.direction == (0, BLOCK_SIZE),   # down
            self.direction == (-BLOCK_SIZE, 0),  # left
            self.direction == (BLOCK_SIZE, 0),   # right
            *food_dir
        ]
        return np.array(state, dtype=int)

# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# --- Replay Memory ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(tuple(args))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        return zip(*batch)

    def __len__(self):
        return len(self.memory)

# --- Evaluation ---
def evaluate(model, episodes=3):
    """Run the trained model greedily and render the game."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    for ep in range(episodes):
        game = SnakeGame()
        state = game.get_state()
        done = False
        while not game.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            with torch.no_grad():
                q = model(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(q).item()
            state, _, done = game.step(action)

            # render
            screen.fill((0, 0, 0))
            for x, y in game.snake:
                pygame.draw.rect(screen, (0, 255, 0), (x, y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(screen, (255, 0, 0), (*game.food, BLOCK_SIZE, BLOCK_SIZE))
            pygame.display.flip()
            clock.tick(FPS)

        print(f"Eval Episode {ep+1}, Score: {game.score}")

    pygame.quit()

# --- Training ---
def train():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    game = SnakeGame()
    state_dim = len(game.get_state())
    action_dim = 3
    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    memory = ReplayMemory(10000)
    batch_size = 128
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    episodes = 500
    for episode in range(episodes):
        game.reset()
        state = game.get_state()
        total_reward = 0
        while not game.done:
            # Epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(action_dim)
            else:
                with torch.no_grad():
                    q_values = model(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_values).item()

            next_state, reward, done = game.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train
            if len(memory) > batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = model(states).gather(1, actions).squeeze()
                next_q_values = model(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)
                loss = criterion(q_values, targets.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Render every 50 episodes
            if episode % 50 == 0:
                screen.fill((0,0,0))
                for x, y in game.snake:
                    pygame.draw.rect(screen, (0,255,0), (x, y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(screen, (255,0,0), (*game.food, BLOCK_SIZE, BLOCK_SIZE))
                pygame.display.flip()
                clock.tick(FPS)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode+1}/{episodes}, Score: {game.score}, Total Reward: {total_reward}")

    # save trained model
    torch.save(model.state_dict(), "snake_dqn.pth")
    print("Model saved to snake_dqn.pth")

    pygame.quit()

if __name__ == "__main__":
    train()

 # load and watch final agent (greedy)
    game = SnakeGame()
    state_dim = len(game.get_state())
    action_dim = 3
    trained = DQN(state_dim, action_dim)
    trained.load_state_dict(torch.load("snake_dqn.pth", map_location="cpu"))
    evaluate(trained, episodes=5)