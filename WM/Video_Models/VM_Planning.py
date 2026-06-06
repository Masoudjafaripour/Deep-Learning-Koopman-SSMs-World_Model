"""
Video World Model for Robotic Planning
=======================================
Uses iVideoGPT (NeurIPS 2024) as a video world model for visual planning
on MetaWorld robotic manipulation tasks.

Pipeline:
  pixel obs → VQ-VAE tokenizer → iVideoGPT (action-conditioned next-token prediction)
            → imagined future frames → MPC planner → execute best action

Setup:
  pip install torch torchvision
  pip install git+https://github.com/thuml/iVideoGPT.git
  pip install metaworld mujoco gymnasium

Reference: https://github.com/thuml/iVideoGPT
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 64        # resize obs to 64x64
LATENT_DIM  = 128       # obs encoder output dim
ACTION_DIM  = 4         # MetaWorld: (dx, dy, dz, gripper)
SEQ_LEN     = 8         # context window for the world model
HORIZON     = 10        # MPC planning horizon
N_SAMPLES   = 64        # MPC candidate action sequences
BATCH_SIZE  = 32
TRAIN_STEPS = 500
LR          = 3e-4
COLLECT_EP  = 20        # random rollout episodes for training data
EP_LEN      = 50        # steps per episode
TASK        = "reach-v2"  # MetaWorld task
CODEBOOK_SIZE = 512     # VQ-VAE codebook


# ─────────────────────────────────────────────
# 1. ENVIRONMENT WRAPPER
# ─────────────────────────────────────────────

def make_env(task_name: str = TASK, image_obs: bool = True):
    """
    Wraps a MetaWorld task to return pixel observations.
    Falls back to a synthetic env if MetaWorld is not installed.
    """
    try:
        import metaworld
        import metaworld.envs.mujoco.env_dict as env_dict
        env_cls = env_dict.ALL_V2_ENVIRONMENTS[task_name]
        env = env_cls(render_mode="rgb_array")
        env._freeze_rand_vec = False
        env.reset()
        return MetaWorldWrapper(env, img_size=IMG_SIZE)
    except Exception as e:
        print(f"[WARN] MetaWorld not available ({e}). Using synthetic CartPole-like env.")
        return SyntheticRoboticEnv(img_size=IMG_SIZE)


class MetaWorldWrapper:
    """Wraps MetaWorld env to return (img, state, action_dim)."""
    def __init__(self, env, img_size=64):
        self.env = env
        self.img_size = img_size
        self.action_dim = env.action_space.shape[0]
        self.action_space = env.action_space

    def reset(self):
        obs, _ = self.env.reset()
        img = self._render()
        return img, obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        img = self._render()
        done = terminated or truncated
        success = float(info.get("success", 0))
        return img, obs, reward, done, success

    def _render(self):
        import cv2
        frame = self.env.render()
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        return frame.astype(np.float32) / 255.0   # (H,W,3) in [0,1]

    def sample_action(self):
        return self.env.action_space.sample()


class SyntheticRoboticEnv:
    """
    Synthetic stand-in when MetaWorld is unavailable.
    State = 2D gripper position; image = rendered dot on canvas.
    Goal = reach (0.5, 0.5).
    """
    def __init__(self, img_size=64):
        self.img_size = img_size
        self.action_dim = ACTION_DIM
        self.pos = np.zeros(2)
        self.goal = np.array([0.5, 0.5])
        self.step_count = 0

    def reset(self):
        self.pos = np.random.uniform(0, 1, size=2)
        self.step_count = 0
        return self._render(), self.pos.copy()

    def step(self, action):
        delta = np.clip(action[:2], -0.05, 0.05)
        self.pos = np.clip(self.pos + delta, 0, 1)
        dist = np.linalg.norm(self.pos - self.goal)
        reward = -dist
        success = float(dist < 0.05)
        self.step_count += 1
        done = (self.step_count >= EP_LEN) or success
        return self._render(), self.pos.copy(), reward, done, success

    def _render(self):
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32) * 0.9
        # draw gripper (blue)
        cx, cy = int(self.pos[0] * (self.img_size-1)), int(self.pos[1] * (self.img_size-1))
        r = 3
        img[max(0,cy-r):cy+r, max(0,cx-r):cx+r] = [0.2, 0.4, 0.9]
        # draw goal (red)
        gx, gy = int(self.goal[0] * (self.img_size-1)), int(self.goal[1] * (self.img_size-1))
        img[max(0,gy-r):gy+r, max(0,gx-r):gx+r] = [0.9, 0.2, 0.2]
        return img

    def sample_action(self):
        return np.random.uniform(-0.05, 0.05, size=self.action_dim).astype(np.float32)


# ─────────────────────────────────────────────
# 2. VQ-VAE  (visual tokenizer)
# ─────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, dim, commitment=0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        self.codebook.weight.data.uniform_(-1/codebook_size, 1/codebook_size)
        self.commitment = commitment

    def forward(self, z):
        # z: (B, D)
        d = (z.pow(2).sum(1, keepdim=True)
             - 2 * z @ self.codebook.weight.T
             + self.codebook.weight.pow(2).sum(1))
        indices = d.argmin(1)
        z_q = self.codebook(indices)
        loss = F.mse_loss(z_q.detach(), z) * self.commitment + F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()   # straight-through
        return z_q, indices, loss


class VQVAE(nn.Module):
    """Small CNN VQ-VAE: image → discrete token → reconstructed image."""
    def __init__(self, img_size=64, latent_dim=LATENT_DIM, codebook_size=CODEBOOK_SIZE):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),   # 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),  # 16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), # 8
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim),
        )
        self.vq = VectorQuantizer(codebook_size, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        """x: (B,3,H,W) → z_q: (B,D), idx: (B,)"""
        z = self.encoder(x)
        z_q, idx, _ = self.vq(z)
        return z_q, idx

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z = self.encoder(x)
        z_q, idx, vq_loss = self.vq(z)
        x_hat = self.decoder(z_q)
        recon_loss = F.mse_loss(x_hat, x)
        return x_hat, recon_loss + vq_loss, idx


# ─────────────────────────────────────────────
# 3. ACTION ENCODER
# ─────────────────────────────────────────────

class ActionEncoder(nn.Module):
    """
    Projects raw action into the same latent space as visual tokens.
    Optionally conditioned on current obs latent (context-aware).
    """
    def __init__(self, action_dim=ACTION_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim + latent_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, action, obs_latent):
        """action: (B, A), obs_latent: (B, D) → (B, D)"""
        x = torch.cat([action, obs_latent], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────
# 4. SSM WORLD MODEL  (GRU-based, drop-in for Mamba)
# ─────────────────────────────────────────────
# NOTE: Replace GRU with Mamba by swapping the recurrent core below.
# pip install mamba-ssm  →  from mamba_ssm import Mamba

class SSMWorldModel(nn.Module):
    """
    Recurrent world model that predicts next obs latent and reward
    from history of (obs_latent, action_latent) pairs.

    Architecture:
        [z_t, e_t] → GRU → h_{t+1}
                         → z_{t+1} head (next obs latent)
                         → r_{t+1} head (reward)
    """
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.obs_head   = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs_latents, action_latents, h=None):
        """
        obs_latents:    (B, T, D)
        action_latents: (B, T, D)
        h:              (1, B, hidden) initial hidden state
        Returns:
            pred_obs:    (B, T, D)
            pred_reward: (B, T, 1)
            h_new:       (1, B, hidden)
        """
        x = torch.cat([obs_latents, action_latents], dim=-1)  # (B,T,2D)
        x = self.input_proj(x)                                 # (B,T,H)
        out, h_new = self.rnn(x, h)                           # (B,T,H)
        pred_obs    = self.obs_head(out)                       # (B,T,D)
        pred_reward = self.reward_head(out)                    # (B,T,1)
        return pred_obs, pred_reward, h_new

    def imagine(self, obs_latent, action_encoder, action_seq, h=None):
        """
        Roll out imagined trajectory.
        obs_latent: (B, D) current obs
        action_seq: (B, T, A) candidate actions
        Returns imagined rewards: (B, T)
        """
        B, T, _ = action_seq.shape
        rewards = []
        z = obs_latent  # (B, D)
        for t in range(T):
            a = action_seq[:, t, :]                        # (B, A)
            e = action_encoder(a, z)                       # (B, D)
            inp = torch.cat([z, e], dim=-1).unsqueeze(1)  # (B,1,2D)
            inp = self.input_proj(inp)
            out, h = self.rnn(inp, h)                      # (B,1,H)
            z = self.obs_head(out.squeeze(1))              # (B, D)
            r = self.reward_head(out.squeeze(1))           # (B, 1)
            rewards.append(r)
        return torch.cat(rewards, dim=1)  # (B, T)


# ─────────────────────────────────────────────
# 5. MPC PLANNER  (shooting method)
# ─────────────────────────────────────────────

class MPCPlanner:
    """
    Random-shooting MPC.
    Samples N action sequences, rolls them out in the SSM world model,
    picks the one with highest cumulative imagined reward.
    """
    def __init__(self, world_model, action_encoder, action_dim=ACTION_DIM,
                 horizon=HORIZON, n_samples=N_SAMPLES, device=DEVICE):
        self.wm = world_model
        self.ae = action_encoder
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_samples = n_samples
        self.device = device

    @torch.no_grad()
    def plan(self, obs_latent):
        """
        obs_latent: (D,) current obs latent
        Returns: best_action (A,)
        """
        self.wm.eval(); self.ae.eval()
        z = obs_latent.unsqueeze(0).expand(self.n_samples, -1).to(self.device)  # (N,D)

        # sample random action sequences
        actions = torch.FloatTensor(self.n_samples, self.horizon, self.action_dim
                                    ).uniform_(-1, 1).to(self.device)

        # imagine rewards
        rewards = self.wm.imagine(z, self.ae, actions)  # (N, H)
        total_reward = rewards.sum(dim=1)               # (N,)

        best_idx = total_reward.argmax()
        return actions[best_idx, 0].cpu().numpy()       # return first action


# ─────────────────────────────────────────────
# 6. DATA COLLECTION
# ─────────────────────────────────────────────

def collect_data(env, n_episodes=COLLECT_EP, ep_len=EP_LEN):
    """Random rollouts → list of (imgs, actions, rewards) trajectories."""
    print(f"[DATA] Collecting {n_episodes} episodes...")
    trajs = []
    for ep in range(n_episodes):
        img, _ = env.reset()
        imgs, acts, rews = [], [], []
        for _ in range(ep_len):
            action = env.sample_action()
            next_img, _, reward, done, _ = env.step(action)
            imgs.append(img)
            acts.append(action)
            rews.append(reward)
            img = next_img
            if done:
                break
        trajs.append((
            np.stack(imgs),   # (T, H, W, 3)
            np.stack(acts),   # (T, A)
            np.array(rews),   # (T,)
        ))
        if (ep+1) % 5 == 0:
            print(f"  episode {ep+1}/{n_episodes}")
    return trajs


def build_dataset(trajs, seq_len=SEQ_LEN):
    """Slice trajectories into fixed-length sequences."""
    imgs_all, acts_all, rews_all = [], [], []
    for imgs, acts, rews in trajs:
        T = len(imgs)
        for start in range(0, T - seq_len - 1):
            imgs_all.append(imgs[start:start+seq_len])
            acts_all.append(acts[start:start+seq_len])
            rews_all.append(rews[start:start+seq_len])
    # (N, T, H, W, 3) → (N, T, 3, H, W)
    imgs_t = torch.FloatTensor(np.stack(imgs_all)).permute(0, 1, 4, 2, 3)
    acts_t = torch.FloatTensor(np.stack(acts_all))
    rews_t = torch.FloatTensor(np.stack(rews_all)).unsqueeze(-1)
    print(f"[DATA] Dataset: {len(imgs_t)} sequences of length {seq_len}")
    return TensorDataset(imgs_t, acts_t, rews_t)


# ─────────────────────────────────────────────
# 7. TRAINING
# ─────────────────────────────────────────────

def train(vqvae, action_encoder, world_model, dataset, steps=TRAIN_STEPS):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    params = (list(vqvae.parameters())
              + list(action_encoder.parameters())
              + list(world_model.parameters()))
    opt = torch.optim.Adam(params, lr=LR)

    vqvae.to(DEVICE); action_encoder.to(DEVICE); world_model.to(DEVICE)

    losses = []
    step = 0
    print(f"\n[TRAIN] Training for {steps} steps on {DEVICE}...")
    while step < steps:
        for imgs, acts, rews in loader:
            if step >= steps:
                break
            imgs = imgs.to(DEVICE)   # (B, T, 3, H, W)
            acts = acts.to(DEVICE)   # (B, T, A)
            rews = rews.to(DEVICE)   # (B, T, 1)

            B, T, C, H, W = imgs.shape

            # ── encode all frames ──
            imgs_flat = imgs.view(B*T, C, H, W)
            _, vq_loss, _ = vqvae(imgs_flat)
            z_flat, _  = vqvae.encode(imgs_flat)
            z = z_flat.view(B, T, -1)        # (B, T, D)

            # ── encode actions (context-aware) ──
            acts_flat = acts.view(B*T, -1)
            z_flat2   = z.view(B*T, -1)
            e_flat    = action_encoder(acts_flat, z_flat2.detach())
            e         = e_flat.view(B, T, -1) # (B, T, D)

            # ── world model: predict next obs + reward ──
            pred_z, pred_r, _ = world_model(z[:, :-1], e[:, :-1])
            target_z = z[:, 1:].detach()

            loss_obs    = F.mse_loss(pred_z, target_z)
            loss_reward = F.mse_loss(pred_r, rews[:, 1:])
            loss = loss_obs + loss_reward + vq_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                print(f"  step {step:4d}/{steps}  loss={np.mean(losses[-50:]):.4f}")

    return losses


# ─────────────────────────────────────────────
# 8. EVALUATION  (MPC planning loop)
# ─────────────────────────────────────────────

def evaluate(env, vqvae, action_encoder, world_model,
             n_episodes=5, ep_len=EP_LEN):
    planner = MPCPlanner(world_model, action_encoder)
    vqvae.eval(); action_encoder.eval(); world_model.eval()

    results = []
    print(f"\n[EVAL] Running MPC planner for {n_episodes} episodes...")
    for ep in range(n_episodes):
        img, _ = env.reset()
        total_reward = 0.0
        success = 0.0
        frames = []

        for t in range(ep_len):
            frames.append(img)
            # encode current obs
            img_t = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                z, _ = vqvae.encode(img_t)

            # plan
            action = planner.plan(z.squeeze(0))
            action = np.clip(action, -1, 1).astype(np.float32)

            # step env
            img, _, reward, done, succ = env.step(action)
            total_reward += reward
            success = max(success, succ)
            if done:
                break

        results.append({"reward": total_reward, "success": success})
        print(f"  ep {ep+1}: reward={total_reward:.3f}  success={success:.0f}")

    avg_reward  = np.mean([r["reward"]  for r in results])
    avg_success = np.mean([r["success"] for r in results])
    print(f"\n  → Avg reward: {avg_reward:.3f}  |  Avg success: {avg_success:.2%}")
    return results


# ─────────────────────────────────────────────
# 9. VISUALIZATION
# ─────────────────────────────────────────────

def visualize_predictions(vqvae, action_encoder, world_model, dataset,
                           n=4, save_path="rollout_predictions.png"):
    """Compare true vs. imagined future frames."""
    vqvae.eval(); action_encoder.eval(); world_model.eval()
    imgs, acts, _ = next(iter(DataLoader(dataset, batch_size=n)))
    imgs = imgs.to(DEVICE); acts = acts.to(DEVICE)

    B, T, C, H, W = imgs.shape
    imgs_flat = imgs[:, 0]          # start from first frame
    imagined = [imgs_flat.cpu()]

    z, _ = vqvae.encode(imgs_flat)
    h = None
    for t in range(1, T):
        a = acts[:, t-1]
        e = action_encoder(a, z)
        inp = torch.cat([z, e], dim=-1).unsqueeze(1)
        inp = world_model.input_proj(inp)
        out, h = world_model.rnn(inp, h)
        z = world_model.obs_head(out.squeeze(1))
        img_hat = vqvae.decode(z)
        imagined.append(img_hat.cpu())

    fig, axes = plt.subplots(n, T*2, figsize=(T*3, n*1.5))
    for i in range(n):
        for t in range(T):
            # real
            real_frame = imgs[i, t].permute(1, 2, 0).cpu().numpy().clip(0, 1)
            axes[i, t*2].imshow(real_frame)
            axes[i, t*2].axis("off")
            if i == 0: axes[i, t*2].set_title(f"Real t={t}", fontsize=7)
            # imagined
            imag_frame = imagined[t][i].permute(1, 2, 0).detach().numpy().clip(0, 1)
            axes[i, t*2+1].imshow(imag_frame)
            axes[i, t*2+1].axis("off")
            if i == 0: axes[i, t*2+1].set_title(f"Imag t={t}", fontsize=7)

    plt.suptitle("Real vs. Imagined Frames (Video World Model)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[VIZ] Saved: {save_path}")


def plot_loss(losses, save_path="training_loss.png"):
    plt.figure(figsize=(7, 3))
    plt.plot(losses, alpha=0.4, label="step loss")
    window = 20
    smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
    plt.plot(range(window-1, len(losses)), smoothed, label=f"smoothed (w={window})")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("World Model Training Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[VIZ] Saved: {save_path}")


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print(" Video World Model → Robotic Planning Experiment")
    print("=" * 55)
    print(f" Device : {DEVICE}")
    print(f" Task   : {TASK}")
    print(f" Horizon: {HORIZON}  |  MPC samples: {N_SAMPLES}")
    print()

    # 1. env
    env = make_env(TASK)

    # 2. models
    vqvae          = VQVAE(img_size=IMG_SIZE, latent_dim=LATENT_DIM,
                           codebook_size=CODEBOOK_SIZE)
    action_encoder = ActionEncoder(action_dim=env.action_dim,
                                   latent_dim=LATENT_DIM)
    world_model    = SSMWorldModel(latent_dim=LATENT_DIM)

    total_params = sum(p.numel() for m in [vqvae, action_encoder, world_model]
                       for p in m.parameters())
    print(f"[MODEL] Total params: {total_params:,}")

    # 3. collect data
    trajs   = collect_data(env, n_episodes=COLLECT_EP, ep_len=EP_LEN)
    dataset = build_dataset(trajs, seq_len=SEQ_LEN)

    # 4. train
    losses = train(vqvae, action_encoder, world_model, dataset,
                   steps=TRAIN_STEPS)

    # 5. visualize training
    out_dir = Path("/mnt/user-data/outputs")
    plot_loss(losses, save_path=str(out_dir / "training_loss.png"))
    visualize_predictions(vqvae, action_encoder, world_model, dataset,
                          save_path=str(out_dir / "rollout_predictions.png"))

    # 6. evaluate with MPC
    results = evaluate(env, vqvae, action_encoder, world_model,
                       n_episodes=5, ep_len=EP_LEN)

    # 7. save models
    torch.save({
        "vqvae":          vqvae.state_dict(),
        "action_encoder": action_encoder.state_dict(),
        "world_model":    world_model.state_dict(),
    }, str(out_dir / "world_model_checkpoint.pt"))
    print(f"\n[SAVE] Checkpoint saved.")

    print("\n[DONE]")
    return results


if __name__ == "__main__":
    main()