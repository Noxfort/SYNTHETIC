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
# File: tests/test_traffic_simulator.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import unittest
import datetime
from src.core.traffic_simulator import (
    TrafficSimulator,
    SmallFlowStrategy,
    MediumFlowStrategy,
    LargeFlowStrategy,
    ChaoticFlowStrategy
)

class TestTrafficSimulator(unittest.TestCase):
    def test_small_flow_strategy(self):
        strategy = SmallFlowStrategy()
        sim = TrafficSimulator(strategy)
        ts = datetime.datetime(2026, 6, 1, 8, 0, 0) # Monday 8 AM (Peak hour)
        res = sim.get_ground_truth(ts)
        self.assertIn("vehicle_flow", res)
        self.assertIn("current_speed", res)
        self.assertIn("free_flow_speed", res)
        self.assertEqual(res["free_flow_speed"], 80)
        self.assertTrue(res["vehicle_flow"] >= 0)

    def test_medium_flow_strategy(self):
        strategy = MediumFlowStrategy()
        sim = TrafficSimulator(strategy)
        ts = datetime.datetime(2026, 6, 1, 12, 0, 0) # Monday noon
        res = sim.get_ground_truth(ts)
        self.assertEqual(res["free_flow_speed"], 70)

    def test_large_flow_strategy(self):
        strategy = LargeFlowStrategy()
        sim = TrafficSimulator(strategy)
        ts = datetime.datetime(2026, 6, 1, 18, 0, 0) # Monday 6 PM (Evening peak)
        res = sim.get_ground_truth(ts)
        self.assertEqual(res["free_flow_speed"], 60)

    def test_chaotic_flow_strategy(self):
        strategy = ChaoticFlowStrategy()
        sim = TrafficSimulator(strategy)
        ts = datetime.datetime(2026, 6, 1, 8, 0, 0)
        res = sim.get_ground_truth(ts)
        self.assertEqual(res["free_flow_speed"], 50)
        self.assertTrue(res["vehicle_flow"] > 0)

if __name__ == "__main__":
    unittest.main()
