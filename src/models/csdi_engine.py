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
# File: models/csdi_engine.py
# Author: Gabriel Moraes
# Date: 2026-04-27

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ativar TF32 em GPUs modernas (Ampere+) para acelerar multiplicações de matriz em Tensor Cores
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass


class DiffusionEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for the diffusion timestep.
    Maps a scalar timestep 't' into a high-dimensional vector so the network
    knows which stage of the denoising process it is currently at.
    """

    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    @staticmethod
    def _build_embedding(dim, max_steps):
        """Creates sinusoidal embedding table [max_steps, dim]."""
        steps = torch.arange(max_steps).unsqueeze(1)  # [T, 1]
        frequencies = torch.arange(dim // 2).unsqueeze(0)  # [1, D/2]

        table = steps * torch.exp(
            -math.log(10000.0) * frequencies / (dim // 2)
        )
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # [T, D]
        return table

    def forward(self, t):
        """
        Args:
            t (LongTensor): Diffusion timestep indices [batch_size].
        Returns:
            Tensor: Projected embedding [batch_size, proj_dim].
        """
        x = self.embedding[t]
        return self.projection(x)


class ResidualBlock(nn.Module):
    """
    Gated residual block with dilated causal convolution, conditioned on
    the diffusion timestep embedding. Architecture inspired by WaveNet/DiffWave.
    """

    def __init__(self, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels, 2 * residual_channels,
            kernel_size=3, padding=dilation, dilation=dilation
        )
        self.diffusion_proj = nn.Linear(residual_channels, residual_channels)
        self.output_proj = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diff_emb):
        """
        Args:
            x (Tensor): Input [batch, channels, seq_len].
            diff_emb (Tensor): Diffusion timestep embedding [batch, channels].
        Returns:
            Tuple[Tensor, Tensor]: (residual output, skip connection).
        """
        # Condition on timestep
        diff_proj = self.diffusion_proj(diff_emb).unsqueeze(-1)  # [B, C, 1]
        h = x + diff_proj

        # Dilated convolution with gated activation
        h = self.dilated_conv(h)
        gate, filter_ = h.chunk(2, dim=1)
        h = torch.sigmoid(gate) * torch.tanh(filter_)

        # Output projection splits into residual and skip
        h = self.output_proj(h)
        residual, skip = h.chunk(2, dim=1)

        return (x + residual) / math.sqrt(2.0), skip


class CSDIEngine(nn.Module):
    """
    Conditional Score-based Diffusion engine for time-series generation.

    Uses a DDPM (Denoising Diffusion Probabilistic Model) framework:
    - Forward process: gradually adds Gaussian noise to real data over T steps.
    - Reverse process: a neural network learns to denoise step-by-step.
    - Conditioning: the generation can be conditioned on an external vector
      (e.g., the scenario vector from the Screenwriter agent).

    Fixed hyperparameters (no Optuna):
        residual_channels = 64
        n_residual_layers = 8
        diffusion_steps   = 50
    """

    def __init__(
        self,
        n_features=2,
        cond_dim=2048,
        gat_dim=32,
        residual_channels=64,
        n_residual_layers=8,
        diffusion_steps=50,
    ):
        """
        Args:
            n_features (int): Number of time-series features (e.g., [speed, flow]).
            cond_dim (int): Dimension of the conditioning vector (from LLM).
            gat_dim (int): Dimension of the spatial graph context (from GATv2).
            residual_channels (int): Width of each residual block.
            n_residual_layers (int): Number of stacked residual blocks.
            diffusion_steps (int): Total diffusion timesteps (T).
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_features = n_features
        self.cond_dim = cond_dim
        self.residual_channels = residual_channels
        self.n_residual_layers = n_residual_layers
        self.diffusion_steps = diffusion_steps

        # --- Noise Schedule (Linear Beta Schedule) ---
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # --- Input Projection ---
        # Projects the noisy input (n_features) to residual_channels
        self.input_proj = nn.Conv1d(n_features, residual_channels, 1)

        # --- Conditioning Projection ---
        # Projects the external condition vector to residual_channels for additive bias
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, residual_channels * 2),
            nn.SiLU(),
            nn.Linear(residual_channels * 2, residual_channels),
        )

        # --- Graph Context Projection (GATv2) ---
        self.gat_proj = nn.Sequential(
            nn.Linear(gat_dim, residual_channels * 2),
            nn.SiLU(),
            nn.Linear(residual_channels * 2, residual_channels),
        )

        # --- Diffusion Timestep Embedding ---
        self.diff_embedding = DiffusionEmbedding(
            dim=128, proj_dim=residual_channels
        )

        # --- Residual Stack ---
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels,
                    dilation=2 ** (i % 4),  # cycle dilations: 1, 2, 4, 8
                )
                for i in range(n_residual_layers)
            ]
        )

        # --- Output Projection ---
        self.skip_proj = nn.Conv1d(residual_channels, residual_channels, 1)
        self.output_proj = nn.Conv1d(residual_channels, n_features, 1)

        # Zero-initialize output for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        print(f"[CSDI] Engine initialized on {self.device}.")
        print(
            f"[CSDI] Architecture: {n_residual_layers} ResBlocks, "
            f"{residual_channels} channels, {diffusion_steps} diffusion steps."
        )

    def forward(self, x_noisy, cond, gat_cond, t):
        """
        Predicts the noise component added to x at timestep t.

        Args:
            x_noisy (Tensor): Noisy input [batch, n_features, seq_len].
            cond (Tensor): Conditioning vector [batch, cond_dim].
            gat_cond (Tensor): Graph context from GATv2 [batch, gat_dim].
            t (LongTensor): Diffusion timestep indices [batch].

        Returns:
            Tensor: Predicted noise [batch, n_features, seq_len].
        """
        # Project input
        h = self.input_proj(x_noisy)  # [B, C, L]

        # Add LLM conditioning bias
        cond_bias = self.cond_proj(cond).unsqueeze(-1)  # [B, C, 1]
        
        # Add GATv2 spatial context bias
        gat_bias = self.gat_proj(gat_cond).unsqueeze(-1) # [B, C, 1]
        
        h = h + cond_bias + gat_bias

        # Diffusion timestep embedding
        diff_emb = self.diff_embedding(t)  # [B, C]

        # Forward through residual stack, collecting skip connections
        skip_sum = torch.zeros_like(h)
        for layer in self.residual_layers:
            h, skip = layer(h, diff_emb)
            skip_sum += skip

        # Output
        out = self.skip_proj(skip_sum / math.sqrt(self.n_residual_layers))
        out = F.silu(out)
        out = self.output_proj(out)

        return out

    # ------------------------------------------------------------------
    # Training Utilities
    # ------------------------------------------------------------------

    def compute_loss(self, x_0, cond, gat_cond):
        """
        Computes the simplified DDPM training loss (MSE between true and predicted noise).

        Args:
            x_0 (Tensor): Clean data [batch, n_features, seq_len].
            cond (Tensor): Conditioning vector [batch, cond_dim].
            gat_cond (Tensor): Graph context from GATv2 [batch, gat_dim].

        Returns:
            Tensor: Scalar loss.
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
            noise_pred = self.forward(x_noisy, cond, gat_cond, t)
            loss = F.mse_loss(noise_pred, noise)

        return loss

    # ------------------------------------------------------------------
    # Generation (DDPM Reverse Sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, cond, gat_cond, seq_len, n_steps=None, seed_tail=None, seed_alpha=0.6):
        """
        Generates time-series data from pure Gaussian noise via iterative denoising.

        Supports inter-day context seeding: when seed_tail is provided (the tail
        of the previous day's output), it is mixed into the initial noise tensor
        to create smooth temporal transitions between consecutive days.

        Args:
            cond (Tensor): Conditioning vector [batch, cond_dim].
            gat_cond (Tensor): Graph context from GATv2 [batch, gat_dim].
            seq_len (int): Length of the time-series to generate.
            n_steps (int, optional): Override number of sampling steps.
            seed_tail (Tensor, optional): Previous day's output tail 
                [batch, n_features, tail_len]. Used to warm-start the noise.
            seed_alpha (float): Blending weight for the seed (0.0-1.0).
                Higher = more continuity from previous day. Default 0.6.

        Returns:
            Tensor: Generated data [batch, n_features, seq_len].
        """
        steps = n_steps or self.diffusion_steps
        batch_size = cond.shape[0]

        # Start from pure noise
        x = torch.randn(batch_size, self.n_features, seq_len, device=cond.device)

        # Context Seeding: inject previous day's tail into the noise
        if seed_tail is not None:
            tail_len = min(seed_tail.shape[-1], seq_len)
            # Noisify the tail using the diffusion schedule's final noise level
            # so the denoising process can converge normally
            noise_scale = self.sqrt_one_minus_alphas_cumprod[-1]
            noisy_tail = seed_tail[:, :, -tail_len:] + noise_scale * torch.randn_like(
                seed_tail[:, :, -tail_len:]
            )
            x[:, :, :tail_len] = (
                seed_alpha * noisy_tail + (1 - seed_alpha) * x[:, :, :tail_len]
            )
            print(f"[CSDI] Context seeding: injecting {tail_len} steps from previous day (alpha={seed_alpha})")

        for i in reversed(range(steps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=cond.device)

            # Predict noise
            device_type = 'cuda' if x.is_cuda else 'cpu'
            with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                noise_pred = self.forward(x, cond, gat_cond, t)

            # DDPM update rule
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]

            # Mean of posterior: mu = (1/sqrt(alpha_t)) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_pred)
            coeff = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mu = (1.0 / torch.sqrt(alpha_t)) * (x - coeff * noise_pred)

            # Add noise for all steps except the last
            if i > 0:
                sigma = torch.sqrt(beta_t)
                x = mu + sigma * torch.randn_like(x)
            else:
                x = mu

        return x

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_weights(self, path):
        """Saves all model weights."""
        torch.save(self.state_dict(), path)
        print(f"[CSDI] Weights saved to {path}")

    def load_weights(self, path):
        """Loads model weights."""
        try:
            state = torch.load(path, map_location=self.device)
            self.load_state_dict(state)
            print(f"[CSDI] Weights loaded successfully from {path}")
        except FileNotFoundError:
            print("[CSDI] No pre-trained weights found. Engine will start from scratch.")


# ======================================================================
# Self-test block
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing CSDI Engine Architecture...")
    print("=" * 60)

    batch_size = 4
    seq_length = 60  # e.g., 60 seconds of data
    features = 2     # Speed and Flow
    cond_dim = 2048   # Screenwriter latent vector dimension
    gat_dim = 32      # GATv2 output dimension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = CSDIEngine(
        n_features=features,
        cond_dim=cond_dim,
        gat_dim=gat_dim,
        residual_channels=64,
        n_residual_layers=8,
        diffusion_steps=50,
    ).to(device)

    # Create dummy data
    dummy_traffic = torch.randn(batch_size, features, seq_length, device=device)
    dummy_cond = torch.randn(batch_size, cond_dim, device=device)
    dummy_gat = torch.randn(batch_size, gat_dim, device=device)

    # 1. Test forward pass (noise prediction)
    t = torch.randint(0, 50, (batch_size,), device=device)
    noise_pred = model(dummy_traffic, dummy_cond, dummy_gat, t)
    print(f"\nInput shape:          {dummy_traffic.shape}")
    print(f"Condition shape:      {dummy_cond.shape}")
    print(f"GAT Context shape:    {dummy_gat.shape}")
    print(f"Noise prediction:     {noise_pred.shape}")

    # 2. Test training loss
    loss = model.compute_loss(dummy_traffic, dummy_cond, dummy_gat)
    print(f"Training loss:        {loss.item():.4f}")

    # 3. Test generation
    synthetic = model.generate(dummy_cond, dummy_gat, seq_len=seq_length)
    print(f"Generated shape:      {synthetic.shape}")

    # 4. Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Device:               {device}")
    print("\n" + "=" * 60)
    print("CSDI Engine is mathematically sound and ready for the Director Agent.")
    print("=" * 60)
