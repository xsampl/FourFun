import pygame
import random
import numpy as np
import torch
from collections import deque

# Game constants
WIDTH, HEIGHT = 400, 600
PLATFORM_WIDTH, PLATFORM_HEIGHT = 60, 10
PLAYER_WIDTH, PLAYER_HEIGHT = 30, 30
GRAVITY = 0.5
JUMP_VELOCITY = -10
PLATFORM_GAP = 80

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ML Platformer Jump Game")
clock = pygame.time.Clock()

class Platform:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, PLATFORM_WIDTH, PLATFORM_HEIGHT)

class Player:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH//2, HEIGHT-PLAYER_HEIGHT-10, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.vel_y = 0

    def update(self, action):
        # action: 0=left, 1=right, 2=do nothing, 3=jump
        if action == 0:
            self.rect.x -= 5
        elif action == 1:
            self.rect.x += 5
        elif action == 3 and self.on_ground:
            self.vel_y = JUMP_VELOCITY

        self.vel_y += GRAVITY
        self.rect.y += int(self.vel_y)

        # Boundaries
        self.rect.x = max(0, min(WIDTH-PLAYER_WIDTH, self.rect.x))

    @property
    def on_ground(self):
        return self.vel_y == 0

class PlatformerEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player = Player()
        self.platforms = [Platform(WIDTH//2-PLATFORM_WIDTH//2, HEIGHT-40)]
        self.score = 0
        self.done = False
        self.generate_platforms()
        return self.get_state()

    def generate_platforms(self):
        y = HEIGHT-PLATFORM_GAP
        while y > -HEIGHT:
            x = random.randint(0, WIDTH-PLATFORM_WIDTH)
            self.platforms.append(Platform(x, y))
            y -= PLATFORM_GAP

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}

        self.player.update(action)
        reward = 0

        # Collision with platforms
        for plat in self.platforms:
            if self.player.rect.colliderect(plat.rect) and self.player.vel_y > 0:
                self.player.rect.bottom = plat.rect.top
                self.player.vel_y = 0
                reward = 1

        # Scroll platforms and player up if player is high
        if self.player.rect.y < HEIGHT//2:
            offset = HEIGHT//2 - self.player.rect.y
            self.player.rect.y += offset
            for plat in self.platforms:
                plat.rect.y += offset
            self.score += offset

        # Remove platforms below screen and add new ones
        self.platforms = [p for p in self.platforms if p.rect.y < HEIGHT]
        while len(self.platforms) < HEIGHT // PLATFORM_GAP + 2:
            y = self.platforms[-1].rect.y - PLATFORM_GAP
            x = random.randint(0, WIDTH-PLATFORM_WIDTH)
            self.platforms.append(Platform(x, y))

        # Game over
        if self.player.rect.y > HEIGHT:
            self.done = True
            reward = -100

        return self.get_state(), reward, self.done, {}

    def get_state(self):
        # State: player x/y, velocity, nearest platform x/y
        nearest = min(self.platforms, key=lambda p: abs(p.rect.y - self.player.rect.y) if p.rect.y < self.player.rect.y else HEIGHT)
        return np.array([
            self.player.rect.x / WIDTH,
            self.player.rect.y / HEIGHT,
            self.player.vel_y / 20,
            nearest.rect.x / WIDTH,
            nearest.rect.y / HEIGHT
        ], dtype=np.float32)

    def render(self):
        screen.fill((135, 206, 250))
        for plat in self.platforms:
            pygame.draw.rect(screen, (0, 255, 0), plat.rect)
        pygame.draw.rect(screen, (255, 0, 0), self.player.rect)
        pygame.display.flip()

# Simple DQN agent
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

def train_agent(episodes=500):
    env = PlatformerEnv()
    state_dim = 5
    action_dim = 4
    agent = DQN(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    memory = deque(maxlen=2000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    watch_interval = 25

    def watch_agent(agent, env, episodes=1):
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                with torch.no_grad():
                    qvals = agent(torch.tensor(state).float().unsqueeze(0))
                    action = qvals.argmax().item()
                state, _, done, _ = env.step(action)
                env.render()
                clock.tick(60)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for t in range(1000):
            if random.random() < epsilon:
                action = random.randint(0, action_dim-1)
            else:
                with torch.no_grad():
                    qvals = agent(torch.tensor(state).float().unsqueeze(0))
                    action = qvals.argmax().item()
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states).float()
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards).float()
                next_states = torch.tensor(next_states).float()
                dones = torch.tensor(dones).float()

                qvals = agent(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_qvals = agent(next_states).max(1)[0]
                targets = rewards + gamma * next_qvals * (1 - dones)
                loss = criterion(qvals, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {ep+1}, Score: {env.score}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # Save trained agent
    torch.save(agent.state_dict(), "dqn_platformer.pth")

if __name__ == "__main__":
    train_agent(episodes=500)
    # To watch agent play:
    env = PlatformerEnv()
    agent = DQN(5, 4)
    agent.load_state_dict(torch.load("dqn_platformer.pth"))
    state = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        with torch.no_grad():
            qvals = agent(torch.tensor(state).float().unsqueeze(0))
            action = qvals.argmax().item()
        state, _, done, _ = env.step(action)
        env.render()
        clock.tick(60)
        if done:
            state = env.reset()
pygame.quit()