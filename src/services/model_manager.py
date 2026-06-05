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
# File: model_manager.py
# Author: Gabriel Moraes
# Date: 2026-06-04

import gc
import torch
from typing import Dict, Any, Optional
from src.models.vae_tcn import VAETCN
from src.models.csdi_engine import CSDIBackbone
from src.core.logger import logger

class SequentialModelManager:
    """
    Responsibility: Handle Lazy Loading and Sequential Release of heavy PyTorch models.
    Ensures that multiple VRAM-heavy models do not coexist in memory.
    """
    def __init__(self, device: torch.device):
        self.device = device
        
        self.guardian_params: Dict[str, Any] = {
            'input_dim': 2,
            'seq_len': 60,
            'tcn_channels': [16, 32, 64],
            'latent_dim': 2048,
            'dropout': 0.2,
        }
        self.guardian: Optional[VAETCN] = None
        
        self.csdi_params: Dict[str, Any] = {
            'n_features': 2,
            'cond_dim': 2048,
            'residual_channels': 64,
            'n_residual_layers': 8,
            'diffusion_steps': 50,
        }
        self.csdi: Optional[CSDIBackbone] = None

    def ensure_guardian(self) -> VAETCN:
        if self.guardian is None:
            p = self.guardian_params
            self.guardian = VAETCN(
                input_dim=p['input_dim'],
                seq_len=p['seq_len'],
                tcn_channels=p['tcn_channels'],
                latent_dim=p['latent_dim'],
                dropout=p.get('dropout', 0.2),
            ).to(self.device)
            self.guardian.eval()
            logger.info("[ModelManager] VAE-TCN Guardian loaded into memory.")
        return self.guardian

    def release_guardian(self) -> None:
        if self.guardian is not None:
            del self.guardian
            self.guardian = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[ModelManager] VAE-TCN Guardian released from memory.")

    def ensure_csdi(self) -> CSDIBackbone:
        if self.csdi is None:
            self.csdi = CSDIBackbone(**self.csdi_params).to(self.device)
            self.csdi.eval()
            logger.info("[ModelManager] CSDI Backbone loaded into memory.")
        return self.csdi

    def release_csdi(self) -> None:
        if self.csdi is not None:
            del self.csdi
            self.csdi = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[ModelManager] CSDI Backbone released from memory.")

    def update_guardian_params(self, new_params: Dict[str, Any]) -> None:
        self.guardian_params.update(new_params)

    def update_csdi_params(self, new_params: Dict[str, Any]) -> None:
        self.csdi_params.update(new_params)
