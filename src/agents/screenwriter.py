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
# File: agents/screenwriter.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import datetime

class ScreenwriterAgent:
    """
    The Creative Agent of the Synthetic System.
    Translates raw constraints (day of week, time, weather seed, city scale) into a rich,
    semantic prompt for the Cognitive Engine (Qwen).
    Now operates from a MACRO perspective with creative weather interpretation.
    """
    def __init__(self, llm_engine):
        self.llm = llm_engine

    def create_daily_script(self, current_date, constraints: dict) -> dict:
        """
        Drafts the daily scenario prompt and delegates vector synthesis to the LLM.
        """
        # Force day of week to English regardless of system locale
        english_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_of_week = constraints.get('day_of_week', english_days[current_date.weekday()])
        
        start_time = constraints.get('start_time', '00:00:00')
        flow_level = constraints.get('flow_level', 'medium').lower()
        weather_seed = constraints.get('weather', 'Clear')

        # Map the UI flow level (PT-BR or EN) to the Macro City Scale context
        city_mapping = {
            "pequeno": "Small City",
            "médio": "Medium City",
            "grande": "Metropolis",
            "caótico": "Megalopolis",
            "small": "Small City",
            "medium": "Medium City",
            "large": "Metropolis",
            "caotic": "Megalopolis"
        }
        city_size = city_mapping.get(flow_level, "Medium City")

        flow_translation = {
            "pequeno": "low",
            "médio": "medium",
            "grande": "high",
            "caótico": "chaotic",
            "small": "low",
            "medium": "medium",
            "large": "high",
            "caotic": "chaotic"
        }
        english_flow = flow_translation.get(flow_level, "medium")

        print(f"[Screenwriter] Drafting MACRO script for {current_date} ({day_of_week}) at {start_time}...")
        print(f"[Screenwriter] Scale: {city_size} | Weather Seed: {weather_seed} | Target Flow: {english_flow.upper()}")

        # The core persona: Shifted from Micro (Driver) to Macro (Omniscient Traffic Network)
        system_instruction = (
            "You are the Synthetic Screenwriter, an expert macro-level traffic network simulator. "
            "Your job is to describe the systemic physical state of an entire city's traffic grid. "
            "DO NOT write from the perspective of a single driver. Think like an omniscient observer "
            "watching the fluid dynamics of thousands of vehicles. Focus strictly on network-wide variables: "
            "overall density, systemic friction, collective braking waves, and macro-level congestion patterns."
        )

        # The specific context for the current simulation chunk with creative freedom
        previous_weather = constraints.get('previous_weather', None)
        
        transition_context = ""
        if previous_weather:
            if previous_weather == weather_seed:
                transition_context = f"The weather from yesterday ('{previous_weather}') persists into today, carrying over its physical effects on the infrastructure. "
            else:
                transition_context = f"Yesterday the city experienced '{previous_weather}'. Today, this transitions into '{weather_seed}'. Consider how the aftermath of yesterday's weather impacts today's initial conditions. "
        
        user_prompt = (
            f"Context: Today is {day_of_week}. The simulation starts exactly at {start_time} (Midnight). "
            f"The environment is a {city_size} with a general traffic flow classification of '{english_flow.upper()}'. "
            f"{transition_context}"
            f"The initial weather seed for today is '{weather_seed}'. "
            "Treat this weather seed as a starting point. You have the creative, yet realistic, freedom "
            "to imagine exactly how this weather behaves, evolves, or manifests over the hours. "
            "Describe the systemic state of the city's traffic arteries, starting from the eerie calm "
            "of midnight, leading up to the initial build-up of the early morning rush hour. "
            f"Detail exactly how the massive scale of a {city_size} combined with your realistic interpretation "
            f"of the '{weather_seed}' physically impacts the collective movement of the vehicle swarm."
        )

        # Command the LLM to dream the scenario and extract the mathematical vector
        scenario_vector = self.llm.dream_scenario_vector(system_instruction, user_prompt)

        # Return the package to the Maestro
        return {
            "date": str(current_date),
            "metadata": {
                "day_of_week": day_of_week,
                "weather": weather_seed,
                "flow_level": flow_level,
                "city_size": city_size,
                "start_time": start_time
            },
            "scenario_vector": scenario_vector
        }