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
# File: physics_interpreter.py
# Author: Gabriel Moraes
# Date: 2026-06-04

import numpy as np
import torch
from typing import Dict, List
from src.core.constants import (
    DEFAULT_SPEED_CLAMP_MIN,
    DEFAULT_SPEED_CLAMP_MAX,
    CSDI_BASE_FLOW,
    CSDI_FLOW_VARIATION,
    CSDI_BASE_SPEED,
    CSDI_SPEED_VARIATION
)

class TrafficPhysicsInterpreter:
    """
    Responsibility: Interpret latent vectors into traffic properties and scale 
    synthetic physical outputs to real-world dimensions.
    """

    @staticmethod
    def interpret_vector(vector: np.ndarray) -> Dict[str, float]:
        """Heuristic decoder for the latent vector."""
        norm: float = float(np.linalg.norm(vector))
        intensity: float = float(np.clip(norm / 25.0, 0.5, 1.5))
        chaos: float = float(np.std(vector) * 10.0)
        chaos = float(np.clip(chaos, 0.0, 1.0))
        return {"intensity": intensity, "chaos": chaos}

    @staticmethod
    def generate_reference_signal(props: Dict[str, float], steps: int, max_context_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates a reference signal for the TCN-VAE tuner.
        Returns data in TCN format: [Batch, Features, Seq_Len].
        
        The signal is capped at max_context_len * 3 steps
        to keep Optuna trials lightweight without losing representative quality.
        """
        ref_steps: int = min(steps, max_context_len * 3)
        
        base_flow: float = 1.0 * props['intensity']
        base_speed: float = 0.8 / props['intensity']
        
        t: np.ndarray = np.linspace(0, 4 * np.pi, ref_steps)
        noise: np.ndarray = np.random.normal(0, props['chaos'], ref_steps)
        
        flow_pattern: np.ndarray = base_flow * (0.5 * (np.sin(t) + 1)) + (noise * 0.1)
        speed_pattern: np.ndarray = base_speed * (1.0 - (flow_pattern / (base_flow * 2.5)))
        
        data: np.ndarray = np.stack([flow_pattern, speed_pattern], axis=0)
        data = data[np.newaxis, :, :]  # [1, 2, ref_steps]
        return torch.tensor(data, dtype=torch.float32).to(device)

    @staticmethod
    def post_process(raw_data: np.ndarray, props: Dict[str, float]) -> Dict[str, List[int]]:
        """Scales CSDI output to Real World Units."""
        intensity: float = props['intensity']
        
        flow_data: List[int] = []
        speed_data: List[int] = []
        
        for i in range(len(raw_data)):
            val_flow: float = float(raw_data[i][0])
            val_speed: float = float(raw_data[i][1])
            
            # CSDI outputs roughly N(0,1) since it's a diffusion model mapping to standard normal.
            final_flow: int = int((CSDI_BASE_FLOW + val_flow * CSDI_FLOW_VARIATION) * intensity)
            final_speed: int = int(CSDI_BASE_SPEED + val_speed * CSDI_SPEED_VARIATION)
            
            # Clamp to realistic bounds
            final_speed = min(max(final_speed, DEFAULT_SPEED_CLAMP_MIN), DEFAULT_SPEED_CLAMP_MAX)
            final_flow = max(final_flow, 0)

            flow_data.append(final_flow)
            speed_data.append(final_speed)
            
        return {
            "vehicle_flow": flow_data,
            "current_speed": speed_data
        }

    @staticmethod
    def hash_properties(props: Dict[str, float]) -> str:
        """Simple string hash to detect state changes."""
        return f"{round(props['intensity'], 1)}_{round(props['chaos'], 1)}"
