# fast_gridworld_cnn_wm.py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

GRID = 9
CELL = 4
IMG = GRID * CELL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

START = (0, 0)
GOAL = (6, 6)
OBSTACLES = {
    (1, 1), (1, 2), (1, 3),
    (3, 1), (3, 2), (3, 3),
    (4, 5), (5, 5)
}

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


def valid(p):
    i, j = p
    return 0 <= i < GRID and 0 <= j < GRID and p not in OBSTACLES


def step(agent, action):
    di, dj = ACTIONS[action]
    nxt = (agent[0] + di, agent[1] + dj)
    return nxt if valid(nxt) else agent


def render(agent):
    img = torch.zeros(4, IMG, IMG)

    def fill(ch, p):
        i, j = p
        img[ch, i*CELL:(i+1)*CELL, j*CELL:(j+1)*CELL] = 1.0

    for obs in OBSTACLES:
        fill(0, obs)

    fill(1, START)
    fill(2, GOAL)
    fill(3, agent)

    return img


def sample_batch(batch_size):
    cells = [
        (i, j)
        for i in range(GRID)
        for j in range(GRID)
        if valid((i, j))
    ]

    xs, acts, ys = [], [], []

    for _ in range(batch_size):
        agent = random.choice(cells)
        action = random.randint(0, 3)
        nxt = step(agent, action)

        xs.append(render(agent))
        acts.append(action)
        ys.append(render(nxt))

    x = torch.stack(xs).to(DEVICE)
    a = torch.tensor(acts, device=DEVICE)
    y = torch.stack(ys).to(DEVICE)

    return x, a, y


class TinyCNNWorldModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.action_emb = nn.Embedding(4, 16)

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        h = IMG // 2
        self.fc = nn.Linear(32 * h * h + 16, 32 * h * h)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, padding=1),
        )

    def forward(self, obs, action):
        z = self.encoder(obs).flatten(1)
        a = self.action_emb(action)
        z = torch.cat([z, a], dim=1)

        h = IMG // 2
        z = self.fc(z).view(-1, 32, h, h)

        return self.decoder(z)


def loss_fn(logits, target):
    weight = torch.ones_like(target)
    weight[target > 0.5] = 10.0
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def decode_agent(pred):
    agent_map = pred[3]
    scores = torch.zeros(GRID, GRID)

    for i in range(GRID):
        for j in range(GRID):
            patch = agent_map[i*CELL:(i+1)*CELL, j*CELL:(j+1)*CELL]
            scores[i, j] = patch.mean()

    return divmod(scores.argmax().item(), GRID)


def train_model(epochs=10, steps_per_epoch=100, batch_size=32):
    model = TinyCNNWorldModel().to(DEVICE)
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

        print(f"epoch {epoch+1:02d} | loss {total / steps_per_epoch:.5f}")

    return model


@torch.no_grad()
def predict_next(model, agent, action):
    obs = render(agent).unsqueeze(0).to(DEVICE)
    act = torch.tensor([action], device=DEVICE)

    pred = torch.sigmoid(model(obs, act))[0].cpu()
    pred_agent = decode_agent(pred)

    return pred_agent, pred


@torch.no_grad()
def plan_with_model(model, start=START, max_steps=25):
    agent = start
    path = [agent]
    frames = [render(agent)]

    visited = set()

    for _ in range(max_steps):
        if agent == GOAL:
            break

        best = None

        for action in range(4):
            pred_agent, pred_img = predict_next(model, agent, action)

            if pred_agent in visited:
                penalty = 5
            else:
                penalty = 0

            dist = abs(pred_agent[0] - GOAL[0]) + abs(pred_agent[1] - GOAL[1])
            score = dist + penalty

            if best is None or score < best[0]:
                best = (score, action, pred_agent, pred_img)

        _, action, agent, pred_img = best
        visited.add(agent)
        path.append(agent)
        frames.append(pred_img)

    return path, frames


def show(img, title=""):
    rgb = torch.zeros(IMG, IMG, 3)

    rgb[img[0] > 0.5] = torch.tensor([0.4, 0.4, 0.4])
    rgb[img[1] > 0.5] = torch.tensor([0.0, 0.2, 1.0])
    rgb[img[2] > 0.5] = torch.tensor([0.0, 1.0, 0.0])
    rgb[img[3] > 0.5] = torch.tensor([1.0, 0.0, 0.0])

    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")


def plot_rollout(path, frames):
    n = len(frames)
    plt.figure(figsize=(2 * n, 2.5))

    for t, img in enumerate(frames):
        plt.subplot(1, n, t + 1)
        show(img, f"t={t}\n{path[t]}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = train_model(
        epochs=10,
        steps_per_epoch=100,
        batch_size=32
    )

    path, frames = plan_with_model(model)

    print("Predicted path:")
    print(path)

    plot_rollout(path, frames)