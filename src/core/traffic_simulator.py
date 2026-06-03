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
from ui.interfaces import IFlowStrategy

class BaseFlowStrategy(IFlowStrategy):
    """
    Base strategy for standard flow calculations (Small, Medium, Large).
    Provides common peak hour and weekday factoring logic.
    """
    def __init__(self, base_amplitude: int, free_flow_speed: int):
        self.base_amplitude = base_amplitude
        self.free_flow_speed = free_flow_speed

    def _get_peak_hour_factor(self, hour: float) -> float:
        peak_morning = np.exp(-((hour - 8)**2) / (2 * 2**2))
        peak_evening = np.exp(-((hour - 18)**2) / (2 * 2.5**2))
        
        base_level = 0.1
        factor = base_level + 0.9 * max(peak_morning, peak_evening)
        return factor

    def _get_weekday_factor(self, weekday: int) -> float:
        if weekday < 5: return 1.0
        elif weekday == 5: return 0.7
        else: return 0.5

    def get_ground_truth(self, timestamp: datetime.datetime) -> dict:
        hour_factor = self._get_peak_hour_factor(timestamp.hour + timestamp.minute / 60.0)
        day_factor = self._get_weekday_factor(timestamp.weekday())
        
        noise = random.uniform(0.90, 1.10)
        current_flow = self.base_amplitude * hour_factor * day_factor * noise
        vehicle_count = max(0, int(current_flow))

        capacity_factor = 2.2
        congestion_metric = vehicle_count / (self.base_amplitude * capacity_factor)
        congestion_factor = max(0.1, 1.0 - congestion_metric)
        
        current_speed = self.free_flow_speed * congestion_factor * noise
        
        return {
            "vehicle_flow": vehicle_count,
            "current_speed": max(5, int(current_speed)),
            "free_flow_speed": self.free_flow_speed
        }


class SmallFlowStrategy(BaseFlowStrategy):
    def __init__(self):
        super().__init__(base_amplitude=30, free_flow_speed=80)


class MediumFlowStrategy(BaseFlowStrategy):
    def __init__(self):
        super().__init__(base_amplitude=120, free_flow_speed=70)


class LargeFlowStrategy(BaseFlowStrategy):
    def __init__(self):
        super().__init__(base_amplitude=250, free_flow_speed=60)


class ChaoticFlowStrategy(IFlowStrategy):
    """
    Special logic for Chaotic flow, ignoring standard peak factors
    to mimic severe congestion close to maximal capacity.
    """
    def __init__(self):
        self.base_amplitude = 600
        self.free_flow_speed = 50

    def get_ground_truth(self, timestamp: datetime.datetime) -> dict:
        saturation = random.uniform(0.70, 0.90)
        vehicle_count = int(self.base_amplitude * saturation)
        
        fluidity = 1.0 - saturation 
        current_speed = self.free_flow_speed * fluidity
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
    
    def __init__(self, strategy: IFlowStrategy):
        self.strategy = strategy
        print(f"[Cérebro] Inicializado com estratégia: {self.strategy.__class__.__name__}")

    def get_ground_truth(self, timestamp: datetime.datetime) -> dict:
        return self.strategy.get_ground_truth(timestamp)