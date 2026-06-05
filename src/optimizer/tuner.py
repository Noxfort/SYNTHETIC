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

from typing import Dict, Any, Optional
from src.core.logger import logger

class HyperTuner:
    """
    AutoML Orchestrator using Optuna.
    Finds the best TCN-VAE (Physics Guardian) and CSDI (Generator) hyperparameters
    for a specific traffic scenario in a 'Just-in-Time' fashion.
    """

    def __init__(self, n_trials: int = 10) -> None:
        """
        Args:
            n_trials (int): Number of different configurations to test.
        """
        self.n_trials: int = n_trials
        self.best_params: Optional[Dict[str, Any]] = None

    def optimize(self, sample_data: torch.Tensor) -> Dict[str, Any]:
        """
        Runs the optimization process for the TCN-VAE Physics Guardian.
        
        Args:
            sample_data (Tensor): A sample of the target data shape [Batch, Features, Seq_Len]
                                  (TCN format: channels-first).
        
        Returns:
            dict: The best hyperparameters found for the TCN-VAE.
        """
        logger.info(f"[AutoML] Starting TCN-VAE optimization with {self.n_trials} trials...")
        
        study: optuna.study.Study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )

        study.optimize(
            lambda trial: self._objective(trial, sample_data),
            n_trials=self.n_trials
        )

        logger.info("[AutoML] Optimization finished.")
        
        # Check if any trial completed — all may have been pruned
        completed: list = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if completed:
            logger.info(f"[AutoML] Best TCN-VAE parameters: {study.best_params}")
            self.best_params = study.best_params
        else:
            logger.warning("[AutoML] WARNING: All trials were pruned. Using safe default parameters.")
            self.best_params = {
                'n_tcn_layers': 3,
                'tcn_base_channels': 16,
                'latent_dim': 128,
                'dropout': 0.2,
                'lr': 1e-3,
                'beta': 1.0,
            }
        
        return self.best_params

    def _objective(self, trial: optuna.trial.Trial, data: torch.Tensor) -> float:
        """
        The objective function evaluated by Optuna.
        Builds a TCN-VAE, trains it briefly, and returns the total loss
        (reconstruction + KL divergence).
        """
        device: torch.device = data.device
        input_dim: int = data.shape[1]   # n_features (channels)
        seq_len: int = data.shape[2]     # sequence length

        # --- 1. Suggest TCN-VAE Hyperparameters ---
        n_tcn_layers: int = trial.suggest_int("n_tcn_layers", 2, 4)
        tcn_base_channels: int = trial.suggest_int("tcn_base_channels", 16, 64, step=16)
        latent_dim: int = trial.suggest_int("latent_dim", 64, 512, step=64)
        dropout: float = trial.suggest_float("dropout", 0.05, 0.4, step=0.05)
        lr: float = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        beta: float = trial.suggest_float("beta", 0.1, 2.0, step=0.1)

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
        except Exception as e:
            # If the architecture is invalid (e.g., dimensions mismatch), prune
            raise optuna.exceptions.TrialPruned(f"Invalid architecture configuration: {e}") from e

        # --- 3. Setup Optimizer ---
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # --- 4. Short Training Loop (5 epochs) ---
        model.train()
        final_loss: float = 0.0
        
        pruner = PruningCallback(trial, monitor="loss")
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        for epoch in range(5):
            optimizer.zero_grad()
            
            device_type: str = 'cuda' if data.is_cuda else 'cpu'
            with torch.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                reconstruction, mu, logvar = model(data)
                total_loss, recon_loss, kld_loss = calculate_vae_loss(
                    reconstruction, data, mu, logvar, beta=beta
                )
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            final_loss = float(total_loss.item())
            
            # Check if we should prune (stop) this trial
            try:
                pruner.check_pruned(epoch, final_loss)
            except optuna.exceptions.TrialPruned:
                del model
                raise

        # Cleanup
        del model
        return final_loss

    def optimize_csdi(self, data: torch.Tensor, cond: torch.Tensor, gat_cond: torch.Tensor) -> Dict[str, Any]:
        """
        Runs JIT optimization for CSDI Engine hyperparameters.
        """
        logger.info(f"[AutoML] Starting CSDI optimization with {self.n_trials} trials...")
        
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )

        study.optimize(
            lambda trial: self._objective_csdi(trial, data, cond, gat_cond),
            n_trials=self.n_trials
        )

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if completed:
            logger.info(f"[AutoML] Best CSDI parameters: {study.best_params}")
            best_params = study.best_params
        else:
            logger.warning("[AutoML] WARNING: All trials were pruned. Using safe CSDI default parameters.")
            best_params = {
                'residual_channels': 64,
                'n_residual_layers': 8,
                'diffusion_steps': 50,
            }
        
        return best_params

    def _objective_csdi(self, trial: optuna.trial.Trial, data: torch.Tensor, cond: torch.Tensor, gat_cond: torch.Tensor) -> float:
        device = data.device
        
        # Suggest CSDI Hyperparameters
        residual_channels = trial.suggest_int("residual_channels", 32, 128, step=32)
        n_residual_layers = trial.suggest_int("n_residual_layers", 4, 12, step=2)
        diffusion_steps = trial.suggest_int("diffusion_steps", 20, 100, step=10)

        try:
            from src.models.csdi_engine import CSDIBackbone
            from src.algorithms.diffusion_process import DiffusionSampler
            model = CSDIBackbone(
                n_features=data.shape[1],
                cond_dim=cond.shape[1],
                gat_dim=gat_cond.shape[1],
                residual_channels=residual_channels,
                n_residual_layers=n_residual_layers,
                diffusion_steps=diffusion_steps
            ).to(device)
            sampler = DiffusionSampler(model, diffusion_steps=diffusion_steps)
        except Exception as e:
            raise optuna.exceptions.TrialPruned(f"Invalid CSDI architecture: {e}") from e

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        final_loss = 0.0
        
        pruner = PruningCallback(trial, monitor="loss")
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        for epoch in range(5):
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
                loss = sampler.compute_loss(data, cond, gat_cond)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            final_loss = float(loss.item())
            
            try:
                pruner.check_pruned(epoch, final_loss)
            except optuna.exceptions.TrialPruned:
                del model
                raise

        del model
        return final_loss