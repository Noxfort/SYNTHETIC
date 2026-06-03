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
# File: models/vae_tcn.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ativar TF32 em GPUs modernas (Ampere+) para acelerar multiplicações de matriz em Tensor Cores
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass

class Chomp1d(nn.Module):
    """
    Removes the extra padding on the right side of the sequence to maintain causality.
    Ensures that output at time t only depends on inputs from time t and earlier.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    A single Temporal Convolutional Network block with residual connection.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks forming the TCN architecture.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                        dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                                        dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class VAETCN(nn.Module):
    """
    Variational Autoencoder using Temporal Convolutional Networks.
    Generates stochastic, hyper-realistic traffic physics.
    """
    def __init__(self, input_dim, seq_len, tcn_channels, latent_dim, dropout=0.2):
        super(VAETCN, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.tcn_channels = tcn_channels
        
        # Encoder: TCN reduces the time-series feature maps
        self.encoder_tcn = TemporalConvNet(input_dim, tcn_channels, kernel_size=3, dropout=dropout)
        self.flatten_dim = tcn_channels[-1] * seq_len
        
        # Latent Space Projections (Muz and LogVar)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder: Maps latent back to the sequence shape, then TCN reconstruction
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        # Reverse the channels for the decoder
        decoder_channels = tcn_channels[::-1]
        self.decoder_tcn = TemporalConvNet(decoder_channels[0], decoder_channels + [input_dim], kernel_size=3, dropout=dropout)

    def encode(self, x):
        """
        Compresses the time-series into a probabilistic distribution.
        Input x shape: (batch_size, input_dim, seq_len)
        """
        enc_out = self.encoder_tcn(x)
        enc_out = enc_out.view(enc_out.size(0), -1) # Flatten
        mu = self.fc_mu(enc_out)
        logvar = self.fc_logvar(enc_out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick: Allows backpropagation through the stochastic node.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Reconstructs the time-series from the latent vector.
        Input z shape: (batch_size, latent_dim)
        """
        dec_in = self.fc_decode(z)
        dec_in = dec_in.view(dec_in.size(0), self.tcn_channels[-1], self.seq_len)
        reconstruction = self.decoder_tcn(dec_in)
        return reconstruction

    def forward(self, x):
        """
        Full pass for training.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def generate_synthetic_data(self, z=None, batch_size=1, device='cpu'):
        """
        Generates brand new, stochastic traffic patterns from pure noise or a conditioned vector.
        """
        if z is None:
            z = torch.randn(batch_size, self.latent_dim).to(device)
        
        device_type = 'cuda' if z.is_cuda else 'cpu'
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                synthetic_seq = self.decode(z)
            
        return synthetic_seq

def calculate_vae_loss(reconstruction, original, mu, logvar, beta=1.0):
    """
    Computes the VAE Loss: Reconstruction Loss (MSE) + KL Divergence.
    Beta controls the weight of the stochasticity.
    """
    recon_loss = F.mse_loss(reconstruction, original, reduction='mean')
    
    # KL Divergence: How closely the latent distribution matches a standard Normal distribution
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss

# Self-test block
if __name__ == "__main__":
    print("Testing VAE-TCN Architecture...")
    batch_size = 4
    seq_length = 60    # e.g., 60 seconds of data
    features = 2       # e.g., Speed and Density
    
    # Create dummy traffic data: (batch, channels, length)
    dummy_traffic = torch.randn(batch_size, features, seq_length)
    
    # Initialize model
    model = VAETCN(input_dim=features, seq_len=seq_length, tcn_channels=[16, 32, 64], latent_dim=128)
    
    # Forward pass
    reconstruction, mu, logvar = model(dummy_traffic)
    loss, recon, kld = calculate_vae_loss(reconstruction, dummy_traffic, mu, logvar)
    
    print(f"Original shape: {dummy_traffic.shape}")
    print(f"Reconstructed shape: {reconstruction.shape}")
    print(f"Latent vector (mu) shape: {mu.shape}")
    print(f"Total Loss: {loss.item():.4f} (Recon: {recon.item():.4f}, KLD: {kld.item():.4f})")
    print("\nArchitecture is mathematically sound and ready for the Director Agent.")