import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def log_normalize(tensor):
    return torch.log1p(tensor)

def log_denormalize(tensor):
    return torch.expm1(tensor)

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

class FactorDiffusionModel(nn.Module):
    def __init__(self, rank, t_emb_dim=64):
        super(FactorDiffusionModel, self).__init__()
        self.rank = rank
        self.t_emb_dim = t_emb_dim

        self.denoiser_A = nn.Sequential(
            nn.Linear(51 * rank + t_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 51 * rank),
            nn.Sigmoid()
        )
        self.denoiser_B = nn.Sequential(
            nn.Linear(51 * rank + t_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 51 * rank),
            nn.Sigmoid()
        )
        self.denoiser_C = nn.Sequential(
            nn.Linear(25 * rank + t_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 25 * rank),
            nn.Sigmoid()
        )

    def forward(self, A, B, C, t):
        batch_size = A.shape[0]
        t_embed = get_timestep_embedding(t.view(batch_size, 1), self.t_emb_dim)

        A = A.view(batch_size, -1)
        B = B.view(batch_size, -1)
        C = C.view(batch_size, -1)

        A_input = torch.cat([A, t_embed], dim=1)
        B_input = torch.cat([B, t_embed], dim=1)
        C_input = torch.cat([C, t_embed], dim=1)

        A_out = self.denoiser_A(A_input).view(batch_size, 51, self.rank)
        B_out = self.denoiser_B(B_input).view(batch_size, 51, self.rank)
        C_out = self.denoiser_C(C_input).view(batch_size, 25, self.rank)
        return A_out, B_out, C_out

def add_scaled_noise(x, t_idx, alpha_bars):
   noise = torch.randn_like(x)
   alpha_bar_t = alpha_bars[t_idx].view(-1, 1, 1)

   return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise


def train_diffusion(model, factors, batch_size=32, lr=1e-3, epochs=100, T=1000, device='cuda:0'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    alpha_bars = make_ddim_schedule(T).to(device)

    A_list, B_list, C_list = [], [], []
    for A, B, C in factors:
        A_list.append(log_normalize(A))
        B_list.append(log_normalize(B))
        C_list.append(log_normalize(C))

    A_tensor = torch.stack(A_list).to(device)
    B_tensor = torch.stack(B_list).to(device)
    C_tensor = torch.stack(C_list).to(device)

    for epoch in range(epochs):
        perm = torch.randperm(A_tensor.size(0))
        A_tensor, B_tensor, C_tensor = A_tensor[perm], B_tensor[perm], C_tensor[perm]
        total_loss = 0

        for i in range(0, A_tensor.size(0), batch_size):
            A_batch = A_tensor[i:i+batch_size]
            B_batch = B_tensor[i:i+batch_size]
            C_batch = C_tensor[i:i+batch_size]

            t_idx = torch.randint(0, T, (A_batch.size(0),), device=device)

            A_noisy = add_scaled_noise(A_batch, t_idx, alpha_bars)
            B_noisy = add_scaled_noise(B_batch, t_idx, alpha_bars)
            C_noisy = add_scaled_noise(C_batch, t_idx, alpha_bars)

            t = alpha_bars[t_idx].view(-1, 1, 1)

            A_pred, B_pred, C_pred = model(A_noisy, B_noisy, C_noisy, t)
            loss = mse_loss(A_pred, A_batch) + mse_loss(B_pred, B_batch) + mse_loss(C_pred, C_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            del A_batch, B_batch, C_batch, A_noisy, B_noisy, C_noisy, A_pred, B_pred, C_pred, t, t_idx, loss

        print(f"Epoch {epoch+1}, Loss: {total_loss / max(1, A_tensor.size(0) // batch_size)}")

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ddim_step(x_t, x0_pred, alpha_t, alpha_prev):
    return (
        torch.sqrt(alpha_prev) * x0_pred +
        torch.sqrt(1 - alpha_prev) * ((x_t - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(1 - alpha_t))
    )

def sample_factors(model, num_samples, rank, T=100, device='cuda:0'):
    model.eval()
    alpha_bars = make_ddim_schedule(T).to(device)

    A = torch.randn(num_samples, 51, rank, device=device)
    B = torch.randn(num_samples, 51, rank, device=device)
    C = torch.randn(num_samples, 25, rank, device=device)

    for i in reversed(range(1, T)):
        alpha_t = alpha_bars[i]
        alpha_prev = alpha_bars[i-1]
        t_tensor = torch.full((num_samples, 1, 1), alpha_t.item(), device=device)

        A0_pred, B0_pred, C0_pred = model(A, B, C, t_tensor)

        A = ddim_step(A, A0_pred, alpha_t, alpha_prev)
        B = ddim_step(B, B0_pred, alpha_t, alpha_prev)
        C = ddim_step(C, C0_pred, alpha_t, alpha_prev)

    return log_denormalize(A0_pred), log_denormalize(B0_pred), log_denormalize(C0_pred)
