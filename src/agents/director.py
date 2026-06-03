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

import gc
import torch
import torch.nn.functional as F
import numpy as np
from src.optimizer.tuner import HyperTuner
from src.models.csdi_engine import CSDIEngine
from src.models.vae_tcn import VAETCN

class DirectorAgent:
    """
    The Tactical Agent of the Synthetic System (Physics-Guarded Edition).
    
    Responsibilities:
    1. Manages Temporal Inertia (blending today's dream with yesterday's reality).
    2. Validates the Dream Vector using the VAE-TCN Physics Guardian with AMP.
    3. Interprets the cleaned vector into physical parameters.
    4. Generates time-series data via CSDI (Conditional Score-based Diffusion).
    
    Resource Strategy (Sequential Release):
    - Only one heavy model is loaded at a time.
    - VAE-TCN loads → validates → releases before CSDI loads.
    - CSDI loads → generates → releases after finishing.
    - This prevents simultaneous memory consumption from multiple models.
    
    Optimization Strategy:
    - The VAE-TCN Guardian is tuned Just-in-Time by Optuna when the scenario changes.
    - The CSDI Generator uses fixed hyperparameters (no Optuna).
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()
        
        # --- 1. The Physics Guardian (VAE-TCN) — Lazy Loaded ---
        # Stores params for on-demand creation; model is NOT loaded at init.
        self._guardian_params = {
            'input_dim': 2,
            'seq_len': 60,
            'tcn_channels': [16, 32, 64],
            'latent_dim': 2048,
            'dropout': 0.2,
        }
        self.guardian = None  # Lazy: loaded only when needed
        
        # --- 2. The Generator (CSDI) — Lazy Loaded ---
        # Fixed hyperparameters stored for on-demand creation.
        self._csdi_params = {
            'n_features': 2,
            'cond_dim': 2048,
            'residual_channels': 64,
            'n_residual_layers': 8,
            'diffusion_steps': 50,
        }
        self.csdi = None  # Lazy: loaded only when needed
        
        # --- 3. The Optimizer (AutoML - TCN-VAE Only) ---
        self.tuner = HyperTuner(n_trials=5) 
        
        # --- 4. State Memory (Temporal Inertia) ---
        self.last_day_vector = None
        
        # --- 5. Temporal Context (Inter-Day CSDI Seeding) ---
        self.last_day_tail = None
        self.tail_context_len = 120  # ~20 minutes of context at 10s interval
        self.last_dream_hash = None

    # ------------------------------------------------------------------
    # Model Lifecycle Management (Lazy Load + Release)
    # ------------------------------------------------------------------

    def _ensure_guardian(self):
        """Loads VAE-TCN into memory if not already loaded."""
        if self.guardian is None:
            p = self._guardian_params
            self.guardian = VAETCN(
                input_dim=p['input_dim'],
                seq_len=p['seq_len'],
                tcn_channels=p['tcn_channels'],
                latent_dim=p['latent_dim'],
                dropout=p.get('dropout', 0.2),
            ).to(self.device)
            self.guardian.eval()
            print("[Director] VAE-TCN Guardian loaded into memory.")

    def _release_guardian(self):
        """Fully releases VAE-TCN from memory. Params are preserved for future rebuild."""
        if self.guardian is not None:
            del self.guardian
            self.guardian = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Director] VAE-TCN Guardian released from memory.")

    def _ensure_csdi(self):
        """Loads CSDI engine into memory if not already loaded."""
        if self.csdi is None:
            self.csdi = CSDIEngine(**self._csdi_params).to(self.device)
            self.csdi.eval()
            print("[Director] CSDI Engine loaded into memory.")

    def _release_csdi(self):
        """Fully releases CSDI from memory."""
        if self.csdi is not None:
            del self.csdi
            self.csdi = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Director] CSDI Engine released from memory.")

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    def action(self, script: dict, duration_steps: int, graph_embedding: torch.Tensor = None) -> dict:
        """
        Executes the generation pipeline with Physics Validation, Inertia,
        and Sequential Resource Release.

        Pipeline order:
        1. Temporal Inertia (no model needed)
        2. VAE-TCN Physics Validation → RELEASE VAE-TCN
        3. CSDI Generation (conditioned on LLM vector + GATv2 map context) → RELEASE CSDI
        4. Post-Processing (no model needed)

        Args:
            script (dict): Must contain 'scenario_vector' from the Screenwriter.
            duration_steps (int): Data points to generate.
            graph_embedding (Tensor, optional): Spatial map context from GATv2 [1, gat_dim].

        Returns:
            dict: The physical data arrays {'vehicle_flow': [], 'current_speed': []}.
        """
        # 1. Ingest the Raw Dream (SLM Output)
        raw_vector = np.array(script['scenario_vector'], dtype=np.float32)
        
        # 2. Apply Temporal Inertia (The "Memory" Step)
        # If we have a history, we blend the new dream with the old reality.
        # Alpha 0.7 means: 70% New Dream, 30% Yesterday's State.
        if self.last_day_vector is not None:
            alpha = 0.7 
            blended_vector = (raw_vector * alpha) + (self.last_day_vector * (1 - alpha))
        else:
            blended_vector = raw_vector

        # ──────────────────────────────────────────────────────────────
        # PHASE A: VAE-TCN (Load → Validate → Retune if needed → Release)
        # ──────────────────────────────────────────────────────────────
        self._ensure_guardian()
        
        # 3. Physics Validation (The "VAE-TCN" Step with AMP)
        blended_tensor = torch.tensor(blended_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            with torch.no_grad():
                physical_projection = self.guardian.decode(blended_tensor)
                clean_mu, _ = self.guardian.encode(physical_projection)
                anomaly_score = F.mse_loss(blended_tensor, clean_mu).item()
                
        clean_vector = clean_mu.squeeze(0).cpu().numpy()
        
        print(f"[Director] Physics Check: Anomaly Score = {anomaly_score:.4f}")
        if anomaly_score > 0.1:
            print("[Director] NOTE: High anomaly detected. The VAE-TCN significantly corrected the SLM's dream.")

        # Update memory for tomorrow
        self.last_day_vector = clean_vector

        # 4. Interpret the Clean Vector
        properties = self._interpret_vector(clean_vector)
        print(f"[Director] Scenario: Intensity={properties['intensity']:.2f}, Chaos={properties['chaos']:.2f}")

        # 5. Check for Retuning (AutoML — TCN-VAE Only)
        current_dream_hash = self._hash_properties(properties)
        
        if self.last_dream_hash != current_dream_hash:
            print(f"[Director] Semantic shift detected. Retuning VAE-TCN Guardian...")
            # Release old guardian before Optuna trials → saves memory for trial models
            self._release_guardian()
            
            reference_data = self._generate_reference_signal(properties, duration_steps)
            best_params = self.tuner.optimize(reference_data)
            
            # Store tuned params for future lazy rebuilds
            n_tcn_layers = best_params['n_tcn_layers']
            tcn_base_channels = best_params['tcn_base_channels']
            tcn_channels = [tcn_base_channels * (2 ** i) for i in range(n_tcn_layers)]
            
            self._guardian_params = {
                'input_dim': 2,
                'seq_len': reference_data.shape[2],
                'tcn_channels': tcn_channels,
                'latent_dim': best_params['latent_dim'],
                'dropout': best_params['dropout'],
            }
            
            del reference_data
            self.last_dream_hash = current_dream_hash

        # Release VAE-TCN — done with physics validation phase
        self._release_guardian()
        del blended_tensor, clean_mu
        
        # ──────────────────────────────────────────────────────────────
        # PHASE B: CSDI (Load → Generate with context seeding → Release)
        # ──────────────────────────────────────────────────────────────
        self._ensure_csdi()
        
        cond_tensor = torch.tensor(clean_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Default graph embedding if none provided
        if graph_embedding is None:
            graph_embedding = torch.zeros((1, 32), dtype=torch.float32, device=self.device)
        else:
            graph_embedding = graph_embedding.to(self.device)
        
        with torch.no_grad():
            raw_output = self.csdi.generate(
                cond_tensor,
                gat_cond=graph_embedding,
                seq_len=duration_steps,
                seed_tail=self.last_day_tail  # Inter-day context (None on day 1)
            )
            
            # Save the tail for the next day's context seeding (kept on CPU to survive release)
            self.last_day_tail = raw_output[:, :, -self.tail_context_len:].cpu().clone()
            
            raw_output = raw_output.squeeze(0).permute(1, 0).cpu().numpy()  # [seq_len, n_features]
        
        # Release CSDI — done with generation phase
        self._release_csdi()
        del cond_tensor
            
        # ──────────────────────────────────────────────────────────────
        # PHASE C: Post-Processing (no models needed)
        # ──────────────────────────────────────────────────────────────
        result = self._post_process(raw_output, properties)
        del raw_output
        
        return result

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _interpret_vector(self, vector):
        """Heuristic decoder for the latent vector."""
        norm = np.linalg.norm(vector)
        intensity = np.clip(norm / 25.0, 0.5, 1.5)
        chaos = np.std(vector) * 10.0
        chaos = np.clip(chaos, 0.0, 1.0)
        return {"intensity": intensity, "chaos": chaos}

    def _generate_reference_signal(self, props, steps):
        """
        Creates a reference signal for the TCN-VAE tuner.
        Returns data in TCN format: [Batch, Features, Seq_Len].
        
        The signal is capped at tail_context_len * 3 steps (~360 at 10s interval)
        to keep Optuna trials lightweight without losing representative quality.
        """
        ref_steps = min(steps, self.tail_context_len * 3)
        
        base_flow = 1.0 * props['intensity']
        base_speed = 0.8 / props['intensity']
        
        t = np.linspace(0, 4*np.pi, ref_steps)
        noise = np.random.normal(0, props['chaos'], ref_steps)
        
        flow_pattern = base_flow * (0.5 * (np.sin(t) + 1)) + (noise * 0.1)
        speed_pattern = base_speed * (1.0 - (flow_pattern / (base_flow * 2.5)))
        
        data = np.stack([flow_pattern, speed_pattern], axis=0)
        data = data[np.newaxis, :, :]  # [1, 2, ref_steps]
        return torch.tensor(data, dtype=torch.float32).to(self.device)

    def _post_process(self, raw_data, props):
        """Scales CSDI output to Real World Units."""
        intensity = props['intensity']
        
        flow_data = []
        speed_data = []
        
        for i in range(len(raw_data)):
            val_flow = raw_data[i][0]
            val_speed = raw_data[i][1]
            
            # CSDI outputs roughly N(0,1) since it's a diffusion model mapping to standard normal.
            # Baseline flow: ~500 veh/hr. Variation: ~300.
            final_flow = int((500 + val_flow * 300) * intensity)
            
            # Baseline speed: ~60 km/h. Variation: ~25.
            final_speed = int(60 + val_speed * 25)
            
            # Clamp to realistic bounds
            final_speed = min(max(final_speed, 20), 110)
            final_flow = max(final_flow, 0)

            flow_data.append(final_flow)
            speed_data.append(final_speed)
            
        return {
            "vehicle_flow": flow_data,
            "current_speed": speed_data
        }

    def _hash_properties(self, props):
        """Simple string hash to detect state changes."""
        return f"{round(props['intensity'], 1)}_{round(props['chaos'], 1)}"