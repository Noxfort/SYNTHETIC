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

# TF32 is now configured globally in main.py


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


class CSDIBackbone(nn.Module):
    """
    Conditional Score-based Diffusion Backbone for time-series generation.
    
    Responsibility: Pure neural network architecture.
    Takes noisy input x_t, conditioning vectors, and timestep t, 
    and predicts the noise added at that step.
    
    Dynamically tuned hyperparameters (Optuna):
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

        print(f"[CSDI] Backbone initialized on {self.device}.")
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
