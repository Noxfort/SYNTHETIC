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
# File: tests/test_environment.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import unittest
import datetime
from src.core.environment import EnvironmentManager

class TestEnvironmentManager(unittest.TestCase):
    def test_get_next_monday_midnight(self):
        next_monday = EnvironmentManager.get_next_monday_midnight()
        self.assertIsInstance(next_monday, datetime.datetime)
        self.assertEqual(next_monday.weekday(), 0) # Monday
        self.assertEqual(next_monday.hour, 0)
        self.assertEqual(next_monday.minute, 0)
        self.assertEqual(next_monday.second, 0)
        self.assertGreater(next_monday, datetime.datetime.now())

    def test_get_dynamic_weather_initial(self):
        state, weather_str = EnvironmentManager.get_dynamic_weather(None)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 3)
        self.assertIsInstance(weather_str, str)
        self.assertTrue(len(weather_str) > 0)

    def test_get_dynamic_weather_transition(self):
        prev_state = ("mild", "Partly Cloudy", "with light wind")
        state, weather_str = EnvironmentManager.get_dynamic_weather(prev_state)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 3)
        self.assertIsInstance(weather_str, str)

if __name__ == "__main__":
    unittest.main()
