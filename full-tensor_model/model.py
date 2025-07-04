import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def log_normalize(tensor):
    return torch.log1p(tensor)

def log_denormalize(tensor):
    return torch.expm1(tensor)

'''
def make_ddim_schedule(T):
    beta_start, beta_end = 1e-4, 0.02
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alpha_bars
'''

def make_ddim_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float32)
    f_t = torch.cos(((steps / T + s) / (1 + s)) * torch.pi * 0.5) ** 2
    alpha_bars = f_t / f_t[0]
    return alpha_bars[1:] 

def get_timestep_embedding(timesteps, embedding_dim=64):
    device = timesteps.device
    half_dim = embedding_dim // 2
    emb = torch.exp(torch.arange(half_dim, device=device) * -np.log(10000) / (half_dim - 1))
    emb = timesteps * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb

class TensorDiffusionModel(nn.Module):
    def __init__(self, tensor_shape=(51, 51, 25), t_emb_dim=64):
        super().__init__()
        self.tensor_shape = tensor_shape
        self.t_emb_dim = t_emb_dim
        self.flat_dim = np.prod(tensor_shape)

        self.denoiser = nn.Sequential(
            nn.Linear(self.flat_dim + t_emb_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(512, self.flat_dim),
            nn.Softplus()
        )

    def forward(self, x, t):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        t_embed = get_timestep_embedding(t.view(batch_size, 1), self.t_emb_dim)
        x_input = torch.cat([x_flat, t_embed], dim=1)
        x_out = self.denoiser(x_input).view(batch_size, *self.tensor_shape)
        return x_out

'''
def add_scaled_noise(x, t, noise_scale=0.1):
    std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
    noise = torch.randn_like(x) * t * std * noise_scale
    return x * (1 - t) + noise
'''

def add_scaled_noise(x, t_idx, alpha_bars):
    noise = torch.randn_like(x)
    alpha_bar_t = alpha_bars[t_idx].view(-1, 1, 1, 1)

    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

    x_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    return x_t

def ddim_step(x_t, x0_pred, alpha_t, alpha_prev):
    return (
        torch.sqrt(alpha_prev) * x0_pred +
        torch.sqrt(1 - alpha_prev) * ((x_t - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(1 - alpha_t))
    )

def train_tensor_diffusion(model, tensor_data, batch_size=32, lr=1e-3, epochs=100, T=1000, device='cuda:1'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    alpha_bars = make_ddim_schedule(T).to(device)

    tensors = [(t) for t in tensor_data]
    tensors = torch.stack(tensors).to(device)

    for epoch in range(epochs):
        perm = torch.randperm(tensors.size(0))
        tensors = tensors[perm]
        total_loss = 0

        model.train()
        for i in range(0, tensors.size(0), batch_size):
            batch = tensors[i:i+batch_size]
            t_idx = torch.randint(0, T, (batch.size(0),), device=device)
            t = alpha_bars[t_idx].view(-1, 1, 1, 1)

            noisy = add_scaled_noise(batch, t_idx, alpha_bars)
            pred = model(noisy, t)
            loss = mse_loss(pred, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            del batch, t_idx, t, noisy, pred, loss
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}, Loss: {total_loss / max(1, tensors.size(0) // batch_size)}")

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sample_tensors(model, num_samples, T=100, device='cuda:1'):
    model.eval()
    tensor_shape = model.tensor_shape
    alpha_bars = make_ddim_schedule(T).to(device)

    x = torch.randn(num_samples, *tensor_shape, device=device)

    for i in reversed(range(1, T)):
        alpha_t = alpha_bars[i]
        alpha_prev = alpha_bars[i-1]
        t_tensor = torch.full((num_samples, 1, 1, 1), alpha_t.item(), device=device)

        x0_pred = model(x, t_tensor)
        x = ddim_step(x, x0_pred, alpha_t, alpha_prev)

    return x0_pred