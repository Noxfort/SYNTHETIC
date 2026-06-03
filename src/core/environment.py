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
# File: core/environment.py
# Author: Gabriel Moraes
# Date: 2026-06-03

import datetime
import random
import json
import os

class EnvironmentManager:
    """
    Manages the physical constraints and passage of time in the simulation.
    Handles the start time and the realistic evolution of atmospheric conditions.
    """
    
    _weather_rules = None

    @classmethod
    def _load_weather_rules(cls):
        if cls._weather_rules is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "weather_rules.json")
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cls._weather_rules = json.load(f)
            except Exception as e:
                print(f"[Environment] Warning: Could not load weather_rules.json: {e}")
                cls._weather_rules = {}
        return cls._weather_rules

    @staticmethod
    def get_next_monday_midnight():
        """
        Calculates the exact datetime for the upcoming Monday at 00:00:00.
        Acts as the 'Ground Zero' for the traffic cycle.
        """
        now = datetime.datetime.now()
        midnight_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        days_ahead = 0 - midnight_today.weekday()
        if days_ahead <= 0: 
            days_ahead += 7
            
        return midnight_today + datetime.timedelta(days=days_ahead)

    @classmethod
    def get_dynamic_weather(cls, weather_state=None):
        """
        Provides a highly realistic dynamic atmospheric seed using a Markov-like state machine.
        Ensures weather evolves logically (e.g., Rain -> Drizzle -> Overcast -> Sunny) and 
        that intensities/characteristics physically match the active condition.
        """
        rules = cls._load_weather_rules()
        
        # Fallbacks in case JSON is missing or corrupted
        transitions = rules.get("transitions", {"Partly Cloudy": ["Partly Cloudy"]})
        category_map = rules.get("category_map", {})
        intensity_map = rules.get("intensity_map", {"Clear/Cloudy": ["mild"], "Precipitation": ["light"], "Extreme": ["severe"]})
        char_map = rules.get("char_map", {})
        
        initial_states = rules.get("initial_states", ["Partly Cloudy"])
        initial_weights = rules.get("initial_weights", [100])
        
        if weather_state is None:
            # Pick a starting state. Bias towards normal weather.
            cond = random.choices(initial_states, weights=initial_weights, k=1)[0]
        else:
            _, prev_cond, _ = weather_state
            if prev_cond in transitions:
                # Evolve condition using the Markov Chain
                cond = random.choice(transitions[prev_cond])
            else:
                cond = "Partly Cloudy"
                
        cat = category_map.get(cond, "Clear/Cloudy")
        intensity = random.choice(intensity_map.get(cat, ["mild"]))
        char = random.choice(char_map.get(cond, [""]))
        
        new_state = (intensity, cond, char)
        
        # Formulate grammatically correct string
        if cat == "Clear/Cloudy":
            # Prevent awkward phrases like "Mild sunny" -> Just "Sunny"
            weather_str = f"{cond} {char}"
        else:
            weather_str = f"{intensity.capitalize()} {cond.lower()} {char}"
            
        return new_state, weather_str
