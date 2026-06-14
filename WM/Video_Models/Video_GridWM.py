# maze_cnn_world_model.py
# Tiny CNN image world model for maze navigation:
# learns (current maze image, action) -> next maze image
# then plans by rolling out predicted next states.

import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
GRID = 7
CELL = 6                  # image size = GRID * CELL = 42x42
IMG = GRID * CELL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

START = (6, 0)
GOAL = (0, 6)

# Actions: up, down, left, right
ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
ACTION_NAMES = ["U", "D", "L", "R"]


# -----------------------------
# Maze walls
# v_walls[r][c] = wall between (r,c) and (r,c+1), c=0..GRID-2
# h_walls[r][c] = wall between (r,c) and (r+1,c), r=0..GRID-2
# -----------------------------
v_walls = torch.zeros(GRID, GRID - 1).bool()
h_walls = torch.zeros(GRID - 1, GRID).bool()

# Hand-made maze, similar style to your image
for r, c in [
    (0, 1), (1, 2), (2, 0), (2, 3), (3, 0), (3, 3),
    (4, 0), (4, 4), (5, 2), (5, 5), (1, 4), (2, 5)
]:
    v_walls[r, c] = True

for r, c in [
    (0, 1), (1, 2), (1, 5), (2, 0), (2, 3), (3, 0),
    (4, 2), (4, 3), (5, 1), (5, 2), (3, 5)
]:
    h_walls[r, c] = True


# -----------------------------
# Maze transition
# -----------------------------
def can_move(pos, action):
    r, c = pos

    if action == 0:  # up
        if r == 0:
            return False
        return not h_walls[r - 1, c]

    if action == 1:  # down
        if r == GRID - 1:
            return False
        return not h_walls[r, c]

    if action == 2:  # left
        if c == 0:
            return False
        return not v_walls[r, c - 1]

    if action == 3:  # right
        if c == GRID - 1:
            return False
        return not v_walls[r, c]

    return False


def step(pos, action):
    if not can_move(pos, action):
        return pos

    dr, dc = ACTIONS[action]
    return (pos[0] + dr, pos[1] + dc)


# -----------------------------
# Rendering as semantic image
# channels:
# 0 = walls
# 1 = start
# 2 = goal
# 3 = agent
# -----------------------------
def render_state(agent):
    img = torch.zeros(4, IMG, IMG)

    # Draw outer border as walls
    img[0, 0:2, :] = 1
    img[0, -2:, :] = 1
    img[0, :, 0:2] = 1
    img[0, :, -2:] = 1

    # Draw vertical internal walls
    for r in range(GRID):
        for c in range(GRID - 1):
            if v_walls[r, c]:
                x = (c + 1) * CELL
                y0 = r * CELL
                y1 = (r + 1) * CELL
                img[0, y0:y1, x - 1:x + 1] = 1

    # Draw horizontal internal walls
    for r in range(GRID - 1):
        for c in range(GRID):
            if h_walls[r, c]:
                y = (r + 1) * CELL
                x0 = c * CELL
                x1 = (c + 1) * CELL
                img[0, y - 1:y + 1, x0:x1] = 1

    def fill(ch, pos):
        r, c = pos
        margin = 2
        y0 = r * CELL + margin
        y1 = (r + 1) * CELL - margin
        x0 = c * CELL + margin
        x1 = (c + 1) * CELL - margin
        img[ch, y0:y1, x0:x1] = 1

    fill(1, START)
    fill(2, GOAL)
    fill(3, agent)

    return img


def decode_agent(pred_img):
    agent_map = pred_img[3]
    scores = torch.zeros(GRID, GRID)

    for r in range(GRID):
        for c in range(GRID):
            y0, y1 = r * CELL, (r + 1) * CELL
            x0, x1 = c * CELL, (c + 1) * CELL
            scores[r, c] = agent_map[y0:y1, x0:x1].mean()

    idx = scores.argmax().item()
    return divmod(idx, GRID)


# -----------------------------
# On-the-fly batch sampling
# -----------------------------
def sample_batch(batch_size):
    xs, acts, ys = [], [], []

    cells = [(r, c) for r in range(GRID) for c in range(GRID)]

    for _ in range(batch_size):
        agent = random.choice(cells)
        action = random.randint(0, 3)
        next_agent = step(agent, action)

        xs.append(render_state(agent))
        acts.append(action)
        ys.append(render_state(next_agent))

    x = torch.stack(xs).to(DEVICE)
    a = torch.tensor(acts, device=DEVICE)
    y = torch.stack(ys).to(DEVICE)

    return x, a, y


# -----------------------------
# Tiny CNN world model
# -----------------------------
class TinyMazeWorldModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.action_emb = nn.Embedding(4, 16)

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        H = IMG // 2

        self.fc = nn.Sequential(
            nn.Linear(32 * H * H + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * H * H),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, padding=1),
        )

    def forward(self, obs, action):
        z = self.encoder(obs).flatten(1)
        a = self.action_emb(action)

        z = torch.cat([z, a], dim=1)
        z = self.fc(z)

        H = IMG // 2
        z = z.view(-1, 32, H, H)

        return self.decoder(z)


def loss_fn(logits, target):
    # rare positive pixels need larger weight
    weight = torch.ones_like(target)
    weight[target > 0.5] = 12.0
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def train_model(epochs=20, steps_per_epoch=150, batch_size=32):
    model = TinyMazeWorldModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total = 0.0

        for _ in range(steps_per_epoch):
            x, a, y = sample_batch(batch_size)

            logits = model(x, a)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"epoch {epoch + 1:02d} | loss {total / steps_per_epoch:.5f}")

    return model


# -----------------------------
# Prediction and planning
# -----------------------------
@torch.no_grad()
def predict_next(model, agent, action):
    obs = render_state(agent).unsqueeze(0).to(DEVICE)
    act = torch.tensor([action], device=DEVICE)

    pred = torch.sigmoid(model(obs, act))[0].cpu()
    pred_agent = decode_agent(pred)

    return pred_agent, pred


@torch.no_grad()
def plan_with_learned_model(model, start=START, goal=GOAL, max_steps=30):
    """
    Greedy planning using learned next-state predictions.
    """
    agent = start
    path = [agent]
    frames = [render_state(agent)]
    visited = set()

    for _ in range(max_steps):
        if agent == goal:
            break

        best = None

        for action in range(4):
            pred_agent, pred_img = predict_next(model, agent, action)

            dist = abs(pred_agent[0] - goal[0]) + abs(pred_agent[1] - goal[1])
            revisit_penalty = 3.0 if pred_agent in visited else 0.0
            stuck_penalty = 1.0 if pred_agent == agent else 0.0

            score = dist + revisit_penalty + stuck_penalty

            if best is None or score < best[0]:
                best = (score, action, pred_agent, pred_img)

        _, action, agent, pred_img = best

        visited.add(agent)
        path.append(agent)
        frames.append(pred_img)

    return path, frames


def bfs_oracle_path(start=START, goal=GOAL):
    """
    Real shortest path using true maze dynamics.
    Only for comparison.
    """
    q = deque([start])
    parent = {start: None}

    while q:
        s = q.popleft()

        if s == goal:
            break

        for a in range(4):
            ns = step(s, a)
            if ns not in parent:
                parent[ns] = s
                q.append(ns)

    if goal not in parent:
        return []

    path = []
    s = goal
    while s is not None:
        path.append(s)
        s = parent[s]

    return path[::-1]


# -----------------------------
# Visualization
# -----------------------------
def to_rgb(img):
    rgb = torch.ones(IMG, IMG, 3)

    # walls black
    rgb[img[0] > 0.5] = torch.tensor([0.0, 0.0, 0.0])

    # start red
    rgb[img[1] > 0.5] = torch.tensor([0.85, 0.1, 0.1])

    # goal green
    rgb[img[2] > 0.5] = torch.tensor([0.0, 0.8, 0.2])

    # agent blue
    rgb[img[3] > 0.5] = torch.tensor([0.1, 0.2, 0.9])

    return rgb


def show_maze_state(img, title=""):
    plt.imshow(to_rgb(img))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")


def plot_path(path, title="Path"):
    img = render_state(path[-1])
    rgb = to_rgb(img)

    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)

    ys = [r * CELL + CELL / 2 for r, c in path]
    xs = [c * CELL + CELL / 2 for r, c in path]

    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_rollout(path, frames, max_show=12):
    n = min(len(frames), max_show)
    plt.figure(figsize=(2.2 * n, 2.5))

    for t in range(n):
        plt.subplot(1, n, t + 1)
        show_maze_state(frames[t], f"t={t}\n{path[t]}")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = train_model(
        epochs=20,
        steps_per_epoch=150,
        batch_size=32,
    )

    learned_path, learned_frames = plan_with_learned_model(model)

    print("\nLearned-model greedy path:")
    print(learned_path)

    oracle = bfs_oracle_path()
    print("\nTrue BFS shortest path:")
    print(oracle)

    plot_rollout(learned_path, learned_frames)
    plot_path(learned_path, "Path from learned world-model planning")
    plot_path(oracle, "Oracle BFS shortest path")