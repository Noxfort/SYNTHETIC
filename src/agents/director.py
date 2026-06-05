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
# File: agents/director.py
# Author: Gabriel Moraes
# Date: 2026-04-27

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from src.algorithms.diffusion_process import DiffusionSampler
from src.optimizer.tuner import HyperTuner
from src.core.logger import logger

from src.services.model_manager import SequentialModelManager
from src.services.physics_interpreter import TrafficPhysicsInterpreter
from src.services.temporal_memory import TemporalContext

class DirectorAgent:
    """
    The Tactical Agent of the Synthetic System (Physics-Guarded Edition).
    
    Responsibilities:
    - Pure Orchestration: Glues together Temporal Memory, Physics Evaluation,
      AutoML Tuning, and CSDI Generation in a sequential pipeline.
    
    Optimization Strategy:
    - The VAE-TCN Guardian is tuned Just-in-Time by Optuna when the scenario changes.
    - The CSDI Generator is also tuned Just-in-Time by Optuna.
    """

    def __init__(self) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp: bool = torch.cuda.is_available()
        
        self.model_manager = SequentialModelManager(self.device)
        self.temporal_ctx = TemporalContext(tail_context_len=120)
        self.tuner = HyperTuner(n_trials=5)
        self.physics = TrafficPhysicsInterpreter()

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    def action(self, script: dict, duration_steps: int, graph_embedding: Optional[torch.Tensor] = None) -> dict:
        """
        Executes the generation pipeline with Physics Validation, Inertia,
        and Sequential Resource Release.
        """
        # 1. Ingest Raw Dream
        raw_vector: np.ndarray = np.array(script['scenario_vector'], dtype=np.float32)
        
        # 2. Apply Temporal Inertia
        blended_vector = self.temporal_ctx.apply_inertia(raw_vector)

        # ──────────────────────────────────────────────────────────────
        # PHASE A: VAE-TCN (Load → Validate → Retune if needed → Release)
        # ──────────────────────────────────────────────────────────────
        guardian = self.model_manager.ensure_guardian()
        blended_tensor = torch.tensor(blended_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            with torch.no_grad():
                physical_projection = guardian.decode(blended_tensor)
                clean_mu, _ = guardian.encode(physical_projection)
                anomaly_score = float(F.mse_loss(blended_tensor, clean_mu).item())
                
        clean_vector = clean_mu.squeeze(0).cpu().numpy()
        
        logger.info(f"[Director] Physics Check: Anomaly Score = {anomaly_score:.4f}")
        if anomaly_score > 0.1:
            logger.info("[Director] NOTE: High anomaly detected. The VAE-TCN significantly corrected the SLM's dream.")

        self.temporal_ctx.update_day_vector(clean_vector)
        properties = self.physics.interpret_vector(clean_vector)
        logger.info(f"[Director] Scenario: Intensity={properties['intensity']:.2f}, Chaos={properties['chaos']:.2f}")

        # 4. Check for Retuning (AutoML)
        current_dream_hash = self.physics.hash_properties(properties)
        
        if self.temporal_ctx.check_shift(current_dream_hash):
            logger.info(f"[Director] Semantic shift detected. Retuning VAE-TCN and CSDI...")
            
            # Release old models before tuning
            self.model_manager.release_guardian()
            self.model_manager.release_csdi()
            
            reference_data = self.physics.generate_reference_signal(
                properties, duration_steps, self.temporal_ctx.tail_context_len, self.device
            )
            
            # Tune VAE-TCN
            best_params_vae = self.tuner.optimize(reference_data)
            n_tcn_layers = int(best_params_vae['n_tcn_layers'])
            tcn_base_channels = int(best_params_vae['tcn_base_channels'])
            
            self.model_manager.update_guardian_params({
                'seq_len': reference_data.shape[2],
                'tcn_channels': [tcn_base_channels * (2 ** i) for i in range(n_tcn_layers)],
                'latent_dim': best_params_vae['latent_dim'],
                'dropout': best_params_vae['dropout'],
            })
            
            # Tune CSDI
            cond_tensor = torch.tensor(clean_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            gat_cond = graph_embedding.to(self.device) if graph_embedding is not None else torch.zeros((1, 32), dtype=torch.float32, device=self.device)
            
            best_params_csdi = self.tuner.optimize_csdi(reference_data, cond_tensor, gat_cond)
            self.model_manager.update_csdi_params(best_params_csdi)
            
            del reference_data
            del cond_tensor
            self.temporal_ctx.update_hash(current_dream_hash)

        # Release VAE-TCN
        self.model_manager.release_guardian()
        del blended_tensor, clean_mu
        
        # ──────────────────────────────────────────────────────────────
        # PHASE B: CSDI (Load → Generate with context seeding → Release)
        # ──────────────────────────────────────────────────────────────
        csdi = self.model_manager.ensure_csdi()
        
        cond_tensor = torch.tensor(clean_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        gat_cond = graph_embedding.to(self.device) if graph_embedding is not None else torch.zeros((1, 32), dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            sampler = DiffusionSampler(csdi, diffusion_steps=csdi.diffusion_steps)
            raw_output = sampler.generate(
                cond_tensor,
                gat_cond=gat_cond,
                seq_len=duration_steps,
                seed_tail=self.temporal_ctx.last_day_tail
            )
            self.temporal_ctx.update_day_tail(raw_output)
            raw_output_np = raw_output.squeeze(0).permute(1, 0).cpu().numpy()
        
        # Release CSDI
        self.model_manager.release_csdi()
        del cond_tensor
            
        # ──────────────────────────────────────────────────────────────
        # PHASE C: Post-Processing (no models needed)
        # ──────────────────────────────────────────────────────────────
        result = self.physics.post_process(raw_output_np, properties)
        del raw_output_np
        
        return result