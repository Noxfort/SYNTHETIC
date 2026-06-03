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
# File: optimizer/tuner.py
# Author: Gabriel Moraes
# Date: 2026-04-27

import optuna
import torch
import torch.optim as optim
from src.models.vae_tcn import VAETCN, calculate_vae_loss
from src.optimizer.callbacks import PruningCallback

class HyperTuner:
    """
    AutoML Orchestrator using Optuna.
    Finds the best TCN-VAE (Physics Guardian) hyperparameters for a specific
    traffic scenario in a 'Just-in-Time' fashion.
    
    NOTE: Optuna optimization is applied ONLY to the TCN-VAE Guardian.
    The CSDI generative engine uses fixed hyperparameters and is not tuned.
    """

    def __init__(self, n_trials=10):
        """
        Args:
            n_trials (int): Number of different configurations to test.
        """
        self.n_trials = n_trials
        self.best_params = None

    def optimize(self, sample_data):
        """
        Runs the optimization process for the TCN-VAE Physics Guardian.
        
        Args:
            sample_data (Tensor): A sample of the target data shape [Batch, Features, Seq_Len]
                                  (TCN format: channels-first).
        
        Returns:
            dict: The best hyperparameters found for the TCN-VAE.
        """
        print(f"[AutoML] Starting TCN-VAE optimization with {self.n_trials} trials...")
        
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )

        study.optimize(
            lambda trial: self._objective(trial, sample_data),
            n_trials=self.n_trials
        )

        print("[AutoML] Optimization finished.")
        
        # Check if any trial completed — all may have been pruned
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if completed:
            print(f"[AutoML] Best TCN-VAE parameters: {study.best_params}")
            self.best_params = study.best_params
        else:
            print("[AutoML] WARNING: All trials were pruned. Using safe default parameters.")
            self.best_params = {
                'n_tcn_layers': 3,
                'tcn_base_channels': 16,
                'latent_dim': 128,
                'dropout': 0.2,
                'lr': 1e-3,
                'beta': 1.0,
            }
        
        return self.best_params

    def _objective(self, trial, data):
        """
        The objective function evaluated by Optuna.
        Builds a TCN-VAE, trains it briefly, and returns the total loss
        (reconstruction + KL divergence).
        """
        device = data.device
        input_dim = data.shape[1]   # n_features (channels)
        seq_len = data.shape[2]     # sequence length

        # --- 1. Suggest TCN-VAE Hyperparameters ---
        n_tcn_layers = trial.suggest_int("n_tcn_layers", 2, 4)
        tcn_base_channels = trial.suggest_int("tcn_base_channels", 16, 64, step=16)
        latent_dim = trial.suggest_int("latent_dim", 64, 512, step=64)
        dropout = trial.suggest_float("dropout", 0.05, 0.4, step=0.05)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        beta = trial.suggest_float("beta", 0.1, 2.0, step=0.1)

        # Build progressive channel list: e.g., [16, 32, 64] for 3 layers
        tcn_channels = [tcn_base_channels * (2 ** i) for i in range(n_tcn_layers)]

        # --- 2. Build the TCN-VAE with suggested params ---
        try:
            model = VAETCN(
                input_dim=input_dim,
                seq_len=seq_len,
                tcn_channels=tcn_channels,
                latent_dim=latent_dim,
                dropout=dropout,
            ).to(device)
        except Exception:
            # If the architecture is invalid (e.g., dimensions mismatch), prune
            raise optuna.exceptions.TrialPruned("Invalid architecture configuration.")

        # --- 3. Setup Optimizer ---
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # --- 4. Short Training Loop (5 epochs) ---
        model.train()
        final_loss = 0.0
        
        pruner = PruningCallback(trial, monitor="loss")
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        for epoch in range(5):
            optimizer.zero_grad()
            
            device_type = 'cuda' if data.is_cuda else 'cpu'
            with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                reconstruction, mu, logvar = model(data)
                total_loss, recon_loss, kld_loss = calculate_vae_loss(
                    reconstruction, data, mu, logvar, beta=beta
                )
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            final_loss = total_loss.item()
            
            # Check if we should prune (stop) this trial
            try:
                pruner.check_pruned(epoch, final_loss)
            except optuna.exceptions.TrialPruned:
                del model
                raise

        # Cleanup
        del model
        return final_loss