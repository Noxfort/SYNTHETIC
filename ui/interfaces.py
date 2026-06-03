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
# File: ui/interfaces.py
# Author: Gabriel Moraes
# Date: 2025-11-27

from abc import ABC, abstractmethod
import datetime

class IDataGenerator(ABC):
    """
    Interface (Abstract Base Class) that all data generators must implement.
    
    This enforces the Liskov Substitution Principle (LSP): any class inheriting
    from this interface can be used interchangeably in the simulation logic
    without breaking the system.
    """

    @abstractmethod
    def generate(self, ground_truth: dict, timestamp: datetime.datetime) -> None:
        """
        Generates the data file based on the ground truth.
        Must be implemented by concrete classes.
        
        Args:
            ground_truth (dict): Dictionary containing physics data (speed, flow).
            timestamp (datetime.datetime): The current simulation time.
        """
        pass

class IFlowStrategy(ABC):
    """
    Interface for traffic flow strategies.
    
    Enforces the Open-Closed Principle (OCP) by allowing new flow categories
    to be added without modifying existing simulator logic.
    """

    @abstractmethod
    def get_ground_truth(self, timestamp: datetime.datetime) -> dict:
        """
        Calculates the ground truth data for the specific flow level.
        
        Args:
            timestamp (datetime.datetime): The current simulation time.
            
        Returns:
            dict: The true traffic physics parameters (flow, speed).
        """
        pass