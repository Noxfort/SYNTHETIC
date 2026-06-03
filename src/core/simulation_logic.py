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
# File: core/simulation_logic.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import datetime
import gc
import json
import math
import random
from typing import List

# --- Core Interfaces and Physics ---
from ui.interfaces import IDataGenerator
from src.core.traffic_simulator import TrafficSimulator, SmallFlowStrategy, MediumFlowStrategy, LargeFlowStrategy, ChaoticFlowStrategy

# --- AI Agents & Engines ---
from src.models.slm_engine import SLMEngine
from src.agents.screenwriter import ScreenwriterAgent
from src.agents.director import DirectorAgent
from src.models.gatv2 import LightweightGATv2
from src.core.environment import EnvironmentManager
import torch

class SimulationOrchestrator:
    """
    The Maestro (AI-Orchestrated).
    Coordinates the Creative and Physics engines with dynamic, non-linear weather
    and strict UI-defined flow constraints. Adheres to SRP by encapsulating logic within
    methods, and DIP by relying on IDataGenerator interface ingestion.
    """

    def __init__(self, config: dict, generators: List[IDataGenerator]):
        self.config = config
        self.generators = generators
        print(f"Starting AI Simulation with configuration:\n{json.dumps(self.config, indent=2, default=str)}")

        slm_mode = self.config['simulation'].get('slm_mode', 'Realista')
        temp_map = {"Ultrarealista": 0.0, "Realista": 0.5, "Criativo": 1.0}
        qwen_temp = temp_map.get(slm_mode, 0.5)
        print(f"[Maestro] Waking up the Neural Engine (Mode: {slm_mode}, Temp: {qwen_temp})...")
        self.slm = SLMEngine(n_gpu_layers=0, temperature=qwen_temp)
        self.screenwriter = ScreenwriterAgent(self.slm)
        self.director = DirectorAgent()

        # Extract map context via GATv2 if a map provider is available
        self.graph_embedding = None
        map_provider = self.config.get('map', {}).get('provider')
        if map_provider:
            print("[Maestro] Extracting Map Topology via GATv2...")
            try:
                self.gatv2 = LightweightGATv2()
                self.graph_embedding = self.gatv2.extract_context(map_provider)
                print("[Maestro] GATv2 Context extracted successfully.")
            except Exception as e:
                print(f"[Maestro] Failed to extract GATv2 context: {e}")

        self._setup_traffic_simulator()

        if not self.generators:
            print("WARNING: No data sources selected. Simulation will run but produce no files.")

    def _setup_traffic_simulator(self):
        """Initializes the traffic brain with the selected flow strategy"""
        flow_level = str(self.config['simulation'].get('flow_level', 'Médio'))
        
        if flow_level == "Pequeno":
            strategy = SmallFlowStrategy()
        elif flow_level == "Grande":
            strategy = LargeFlowStrategy()
        elif flow_level == "Caótico":
            strategy = ChaoticFlowStrategy()
        else:
            strategy = MediumFlowStrategy()
            
        self.traffic_simulator = TrafficSimulator(strategy)
        self.ui_flow_level = flow_level



    def run(self):
        """
        Runs the simulation in two phases for optimal memory usage:
        
        Phase 1 (LLM Active): Pre-generate ALL daily scripts using Qwen,
                               then fully release the LLM from memory (~1.7GB freed).
        Phase 2 (Director Active): Process each day with VAE-TCN + CSDI,
                                    releasing each model after use (sequential release).
        
        This ensures the three heavy models (Qwen, VAE-TCN, CSDI) never coexist in memory.
        """
        try:
            # --- Time Management ---
            sim_config = self.config['simulation']
            start_time = EnvironmentManager.get_next_monday_midnight()
            duration_days = int(sim_config.get('duration_days', 1))
            # Subtract 1 second so the loop strictly bounds up to 23:59:59 of the last day
            end_time = start_time + datetime.timedelta(days=duration_days) - datetime.timedelta(seconds=1)
            interval_seconds = int(sim_config['interval_seconds'])

            print(f"[Maestro] Simulation Loop Started: {start_time} to {end_time}")

            inject_gaps = self.config['problems']['gaps']
            inject_anomalies = self.config['problems']['anomalies']

            # =============================================================
            # PHASE 1: Pre-generate ALL daily scripts (LLM loaded)
            # =============================================================
            print(f"[Maestro] ═══ Phase 1: Dreaming scenarios with LLM ═══")
            
            day_plans = []
            scan_time = start_time
            previous_weather = None
            weather_state = None
            
            while scan_time < end_time:
                next_midnight = (scan_time + datetime.timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                chunk_end = min(next_midnight, end_time)
                
                seconds_in_chunk = (chunk_end - scan_time).total_seconds()
                steps_in_chunk = math.ceil(seconds_in_chunk / interval_seconds)
                
                if steps_in_chunk <= 0:
                    scan_time = next_midnight
                    continue

                current_date = scan_time.date()
                weather_state, current_weather = EnvironmentManager.get_dynamic_weather(weather_state)
                
                constraints = {
                    "weather": current_weather,
                    "previous_weather": previous_weather,
                    "flow_level": self.ui_flow_level, 
                    "day_of_week": current_date.strftime("%A"), 
                    "start_time": "00:00:00"
                }
                
                daily_script = self.screenwriter.create_daily_script(current_date, constraints)
                previous_weather = current_weather
                
                day_plans.append({
                    'script': daily_script,
                    'steps': steps_in_chunk,
                    'start_time': scan_time,
                    'weather': current_weather,
                    'constraints': constraints,
                    'date': current_date,
                })
                
                scan_time = next_midnight
            
            # Release LLM — all scripts generated, free ~1.7GB
            print(f"[Maestro] Phase 1 complete: {len(day_plans)} daily scripts generated.")
            self.slm.release()
            
            # =============================================================
            # PHASE 2: Process each day with Director (VAE-TCN + CSDI)
            # Director handles sequential loading/releasing internally.
            # =============================================================
            print(f"[Maestro] ═══ Phase 2: Processing days with Director ═══")
            
            current_time = start_time
            total_days = len(day_plans)
            
            for day_idx, plan in enumerate(day_plans):
                steps_in_chunk = plan['steps']
                current_weather = plan['weather']
                current_date = plan['date']
                constraints = plan['constraints']
                
                physics_data = self.director.action(plan['script'], steps_in_chunk, self.graph_embedding)
                
                print(f"[Maestro] Generating and Corrupting {steps_in_chunk} files for "
                      f"{current_date} ({constraints['day_of_week']}) | Weather: {current_weather}...")
                
                for step_idx in range(steps_in_chunk):
                    if current_time >= end_time:
                        break
                    
                    self._process_step(
                        step_idx, current_time, current_weather, physics_data,
                        inject_anomalies, inject_gaps
                    )
                    current_time += datetime.timedelta(seconds=interval_seconds)

                # --- Memory Cleanup between days ---
                print(f"[Maestro] Day {day_idx + 1}/{total_days} ({current_date}) completed. Releasing memory...")
                del physics_data
                gc.collect()

            print("[Maestro] Simulation finished successfully!")
            return True, None 

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Maestro] Catastrophic simulation error: {e}")
            return False, e

    def _process_step(self, step_idx, current_time, weather, physics_data, inject_anomalies, inject_gaps):
        """Processes a single step in the simulation."""
        
        # In previously provided code, director's action provided physics_data containing lists
        idx = min(step_idx, len(physics_data.get('current_speed', [])) - 1)
        if idx < 0:
            return  # Safety fallback if physics_data is empty
        
        # --- A. PURE GROUND TRUTH PHYSICS ---
        ground_truth = self.traffic_simulator.get_ground_truth(current_time)
        ground_truth['weather'] = weather

        # --- B. SENSOR FAULT INJECTION (CORRUPTION LAYER) ---
        sensor_data = ground_truth.copy()

        if inject_anomalies:
            if random.random() < 0.05:
                sensor_data['current_speed'] = min(120, sensor_data['current_speed'] * random.uniform(0.3, 1.8))
                sensor_data['vehicle_flow'] = int(sensor_data['vehicle_flow'] * random.uniform(0.1, 4.0))

        if inject_gaps:
            if random.random() < 0.03:
                sensor_data['current_speed'] = None
                sensor_data['vehicle_flow'] = None

        # --- C. GENERATE FINAL FILES ---
        for generator in self.generators:
            generator.generate(sensor_data, current_time)