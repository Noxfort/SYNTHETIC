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
# File: temporal_memory.py
# Author: Gabriel Moraes
# Date: 2026-06-04

import numpy as np
import torch
from typing import Optional
from src.core.constants import TEMPORAL_INERTIA_ALPHA

class TemporalContext:
    """
    Responsibility: Handle temporal inertia math (inter-day blending) 
    and maintain contextual memory of the simulation state.
    """
    def __init__(self, tail_context_len: int = 120):
        self.last_day_vector: Optional[np.ndarray] = None
        self.last_day_tail: Optional[torch.Tensor] = None
        self.last_dream_hash: Optional[str] = None
        self.tail_context_len: int = tail_context_len

    def apply_inertia(self, raw_vector: np.ndarray) -> np.ndarray:
        """Blends the new semantic vector with the previous day's reality."""
        if self.last_day_vector is not None:
            alpha: float = TEMPORAL_INERTIA_ALPHA
            return (raw_vector * alpha) + (self.last_day_vector * (1 - alpha))
        return raw_vector

    def check_shift(self, current_hash: str) -> bool:
        """Checks if a semantic shift occurred compared to the last day."""
        return self.last_dream_hash != current_hash

    def update_hash(self, current_hash: str) -> None:
        self.last_dream_hash = current_hash

    def update_day_vector(self, clean_vector: np.ndarray) -> None:
        self.last_day_vector = clean_vector

    def update_day_tail(self, raw_output: torch.Tensor) -> None:
        """Saves the tail of generated data for next day's context seeding (kept on CPU)."""
        self.last_day_tail = raw_output[:, :, -self.tail_context_len:].cpu().clone()
