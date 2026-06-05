# SYNTHETIC  - An AI-Orchestrated Engine for Multi-Modal Traffic Scenario Synthesis
# Copyright (C) 2026 Noxfort Systems 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SOFTWARE.
#
# File: diffusion_process.py
# Author: Gabriel Moraes
# Date: 2026-06-04

import torch
import torch.nn.functional as F

class DiffusionSampler:
    """
    Responsibility: Handle Denoising Diffusion Probabilistic Models (DDPM) math.
    Encapsulates the noise schedule, loss computation, and generation (reverse sampling).
    """

    def __init__(self, model: torch.nn.Module, diffusion_steps: int = 50):
        self.model = model
        self.diffusion_steps = diffusion_steps
        self.device = next(model.parameters()).device

        # --- Noise Schedule (Linear Beta Schedule) ---
        betas = torch.linspace(1e-4, 0.02, diffusion_steps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def compute_loss(self, x_0: torch.Tensor, cond: torch.Tensor, gat_cond: torch.Tensor) -> torch.Tensor:
        """
        Computes the simplified DDPM training loss (MSE between true and predicted noise).
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=x_0.device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Create noisy version: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        x_noisy = sqrt_alpha * x_0 + sqrt_one_minus * noise

        # Predict noise
        device_type = 'cuda' if x_0.is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
            noise_pred = self.model(x_noisy, cond, gat_cond, t)
            loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def generate(self, cond: torch.Tensor, gat_cond: torch.Tensor, seq_len: int, n_steps: int = None, seed_tail: torch.Tensor = None, seed_alpha: float = 0.6) -> torch.Tensor:
        """
        Generates time-series data from pure Gaussian noise via iterative denoising.
        Supports inter-day context seeding.
        """
        steps = n_steps or self.diffusion_steps
        batch_size = cond.shape[0]

        # Start from pure noise
        x = torch.randn(batch_size, self.model.n_features, seq_len, device=cond.device)

        # Context Seeding: inject previous day's tail into the noise
        if seed_tail is not None:
            tail_len = min(seed_tail.shape[-1], seq_len)
            noise_scale = self.sqrt_one_minus_alphas_cumprod[-1]
            noisy_tail = seed_tail[:, :, -tail_len:] + noise_scale * torch.randn_like(seed_tail[:, :, -tail_len:])
            x[:, :, :tail_len] = (seed_alpha * noisy_tail + (1 - seed_alpha) * x[:, :, :tail_len])
            # print(f"[DiffusionSampler] Context seeding: injecting {tail_len} steps from previous day (alpha={seed_alpha})")

        for i in reversed(range(steps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=cond.device)

            # Predict noise
            device_type = 'cuda' if x.is_cuda else 'cpu'
            with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                noise_pred = self.model(x, cond, gat_cond, t)

            # DDPM update rule
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]

            coeff = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mu = (1.0 / torch.sqrt(alpha_t)) * (x - coeff * noise_pred)

            # Add noise for all steps except the last
            if i > 0:
                sigma = torch.sqrt(beta_t)
                x = mu + sigma * torch.randn_like(x)
            else:
                x = mu

        return x
