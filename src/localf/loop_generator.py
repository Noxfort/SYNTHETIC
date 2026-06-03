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
# File: localf/loop_generator.py
# Author: Gabriel Moraes
# Date: 2026-02-27

import os
import datetime
import random
from ui.interfaces import IDataGenerator

class LoopGenerator(IDataGenerator):
    """
    Generates exclusively Inductive Loop data in CSV format.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = os.path.join(config['output_directory'], 'loop')
        self.problems = config['problems']
        self.sim_config = config['simulation']
        self.loop_locations = config.get('map', {}).get('local_points', {}).get('loops', [])
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load base from loop.json
        self.base_data = self._load_base()
        
        # Quantity configuration
        num_loops = self.sim_config.get('num_loops', 1)
        
        # ID GENERATION
        self.loop_ids = [f"loop_{i:02d}" for i in range(1, num_loops + 1)]

        # Pre-creation of individual folders
        for loop_id in self.loop_ids:
            os.makedirs(os.path.join(self.output_dir, loop_id), exist_ok=True)

    def _load_base(self) -> dict:
        """Loads base structure from src/base/loop.json."""
        import json
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_path, 'src', 'base', 'loop.json')

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[CRITICAL] Could not load loop.json: {e}")
            return {"sensors": [{"id": "loop_01"}], "csv_header": "DATE,TIME,SENSOR_ID,LAT,LON,LANE,CLASS,SPEED_KMH,OCCUPANCY_MS"}

    def generate(self, ground_truth: dict, timestamp: datetime.datetime) -> None:
        """Generates CSV counting files for all configured loops."""
        
        # MAESTRO MACRO-GAP CHECK: Abort generation if the physics engine injected a blackout
        if ground_truth.get('vehicle_flow') is None or ground_truth.get('current_speed') is None:
            return

        for idx, sensor_id in enumerate(self.loop_ids):
            if self.problems['gaps'] and random.random() < 0.10:
                continue

            loop_loc = self.loop_locations[idx] if idx < len(self.loop_locations) else {"lat": 0.0, "lon": 0.0}

            local_volume_factor = random.uniform(0.04, 0.06)
            # Garantir pelo menos 1 veículo (mesmo de madrugada) para o arquivo não ficar vazio
            vehicle_count = max(1, int(ground_truth['vehicle_flow'] * local_volume_factor))
            
            # CSV Header
            csv_content = [self.base_data.get("csv_header", "DATE,TIME,SENSOR_ID,LAT,LON,LANE,CLASS,SPEED_KMH,OCCUPANCY_MS")]
            
            for _ in range(vehicle_count):
                offset = random.randint(0, 5)
                event_time = timestamp + datetime.timedelta(seconds=offset)
                
                date_str = event_time.strftime("%Y-%m-%d")
                time_str = event_time.strftime("%H:%M:%S")
                
                # Lane Logic: 1 (Fast/Light) vs 2 (Slow/Heavy)
                lane = random.choices([1, 2], weights=[0.6, 0.4])[0]
                
                if lane == 1:
                    v_class = random.choice(["LIGHT", "LIGHT", "MOTORCYCLE"])
                    speed = int(ground_truth['current_speed'] * random.uniform(0.95, 1.20))
                else:
                    v_class = random.choice(["LIGHT", "HEAVY", "HEAVY", "BUS"])
                    speed = int(ground_truth['current_speed'] * random.uniform(0.70, 0.95))

                # Physics for Occupancy Time
                length = 4.0 if v_class in ["LIGHT", "MOTORCYCLE"] else 12.0
                speed_ms = max(1, speed / 3.6)
                occupancy = int((length / speed_ms) * 1000)
                
                lat, lon = loop_loc["lat"], loop_loc["lon"]
                line = f"{date_str},{time_str},{sensor_id},{lat},{lon},{lane},{v_class},{speed},{occupancy}"
                csv_content.append(line)

            if self.problems['anomalies'] and random.random() < 0.03:
                csv_content = ["#ERROR: CONTROLLER RESTART", "#DUMP_FAILED"]

            # Save to specific folder: output/loop/loop_XX/file.csv
            ts_string = timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{sensor_id}_{ts_string}.csv"
            sensor_dir = os.path.join(self.output_dir, sensor_id)
            filepath = os.path.join(sensor_dir, filename)
            
            try:
                with open(filepath, 'w') as f:
                    f.write("\n".join(csv_content))
            except Exception as e:
                print(f"LoopGenerator Error {sensor_id}: {e}")