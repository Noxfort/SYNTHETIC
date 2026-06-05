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
# File: core/traffic_simulator.py
# Author: Gabriel Moraes
# Date: 2025-11-27

import numpy as np
import datetime
import random
from typing import Dict, Any
from ui.interfaces import IFlowStrategy
from src.core.logger import logger

class BaseFlowStrategy(IFlowStrategy):
    """
    Base strategy for standard flow calculations (Small, Medium, Large).
    Provides common peak hour and weekday factoring logic.
    """
    def __init__(self, base_amplitude: int, free_flow_speed: int) -> None:
        self.base_amplitude: int = base_amplitude
        self.free_flow_speed: int = free_flow_speed

    def _get_peak_hour_factor(self, hour: float) -> float:
        # Standard morning peak at 8:00, evening peak at 18:00
        peak_morning: float = float(np.exp(-((hour - 8)**2) / (2 * 2**2)))
        peak_evening: float = float(np.exp(-((hour - 18)**2) / (2 * 2.5**2)))
        
        base_level: float = 0.1
        factor: float = base_level + 0.9 * max(peak_morning, peak_evening)
        return factor

    def _get_weekday_factor(self, weekday: int) -> float:
        if weekday < 5: 
            return 1.0
        elif weekday == 5: 
            return 0.7
        else: 
            return 0.5

    def get_ground_truth(self, timestamp: datetime.datetime) -> Dict[str, Any]:
        hour_factor: float = self._get_peak_hour_factor(timestamp.hour + timestamp.minute / 60.0)
        day_factor: float = self._get_weekday_factor(timestamp.weekday())
        
        noise: float = random.uniform(0.90, 1.10)
        current_flow: float = self.base_amplitude * hour_factor * day_factor * noise
        vehicle_count: int = max(0, int(current_flow))

        capacity_factor: float = 2.2
        congestion_metric: float = vehicle_count / (self.base_amplitude * capacity_factor)
        congestion_factor: float = max(0.1, 1.0 - congestion_metric)
        
        current_speed: float = self.free_flow_speed * congestion_factor * noise
        
        return {
            "vehicle_flow": vehicle_count,
            "current_speed": max(5, int(current_speed)),
            "free_flow_speed": self.free_flow_speed
        }


class SmallFlowStrategy(BaseFlowStrategy):
    def __init__(self) -> None:
        super().__init__(base_amplitude=30, free_flow_speed=80)


class MediumFlowStrategy(BaseFlowStrategy):
    def __init__(self) -> None:
        super().__init__(base_amplitude=120, free_flow_speed=70)


class LargeFlowStrategy(BaseFlowStrategy):
    def __init__(self) -> None:
        super().__init__(base_amplitude=250, free_flow_speed=60)


class ChaoticFlowStrategy(IFlowStrategy):
    """
    Special logic for Chaotic flow, ignoring standard peak factors
    to mimic severe congestion close to maximal capacity.
    """
    def __init__(self) -> None:
        self.base_amplitude: int = 600
        self.free_flow_speed: int = 50

    def get_ground_truth(self, timestamp: datetime.datetime) -> Dict[str, Any]:
        saturation: float = random.uniform(0.70, 0.90)
        vehicle_count: int = int(self.base_amplitude * saturation)
        
        fluidity: float = 1.0 - saturation 
        current_speed: float = self.free_flow_speed * fluidity
        current_speed = max(3, int(current_speed))

        return {
            "vehicle_flow": vehicle_count,
            "current_speed": current_speed,
            "free_flow_speed": self.free_flow_speed
        }


class TrafficSimulator:
    """
    The 'Brain' of the simulation context.
    Delegates to a Flow Strategy depending on the level to adhere to OCP.
    """
    
    def __init__(self, strategy: IFlowStrategy) -> None:
        self.strategy: IFlowStrategy = strategy
        logger.info(f"[Cérebro] Inicializado com estratégia: {self.strategy.__class__.__name__}")

    def get_ground_truth(self, timestamp: datetime.datetime) -> Dict[str, Any]:
        return self.strategy.get_ground_truth(timestamp)