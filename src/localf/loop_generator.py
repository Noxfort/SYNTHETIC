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
from typing import Dict, List, Any, Optional
from ui.interfaces import IDataGenerator
from src.core.logger import logger
from src.core.constants import (
    DEFAULT_FALLBACK_LAT,
    DEFAULT_FALLBACK_LON,
    DEFAULT_GAP_PROB_LOOP,
    DEFAULT_ANOMALY_PROB_LOOP,
    LOOP_VOLUME_FACTOR_MIN,
    LOOP_VOLUME_FACTOR_MAX,
    LOOP_LANE_WEIGHTS,
    LOOP_LANE_1_SPEED_MULTIPLIER_MIN,
    LOOP_LANE_1_SPEED_MULTIPLIER_MAX,
    LOOP_LANE_2_SPEED_MULTIPLIER_MIN,
    LOOP_LANE_2_SPEED_MULTIPLIER_MAX,
    LOOP_LIGHT_VEHICLE_LENGTH,
    LOOP_HEAVY_VEHICLE_LENGTH
)

class LoopGenerator(IDataGenerator):
    """
    Generates exclusively Inductive Loop data in CSV format.
    """
    
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.output_dir: str = os.path.join(config['output_directory'], 'loop')
        self.problems: dict = config['problems']
        self.sim_config: dict = config['simulation']
        self.loop_locations: List[Dict[str, float]] = config.get('map', {}).get('local_points', {}).get('loops', [])
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load base from loop.json
        self.base_data: dict = self._load_base()
        
        # Quantity configuration
        num_loops: int = self.sim_config.get('num_loops', 1)
        
        # ID GENERATION
        self.loop_ids: List[str] = [f"loop_{i:02d}" for i in range(1, num_loops + 1)]

        # Pre-creation of individual folders
        for loop_id in self.loop_ids:
            os.makedirs(os.path.join(self.output_dir, loop_id), exist_ok=True)

    def _load_base(self) -> dict:
        """Loads base structure from src/base/loop.json."""
        import json
        base_path: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path: str = os.path.join(base_path, 'src', 'base', 'loop.json')

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"[LoopGenerator] Base schema config not found: {e}")
            return {"sensors": [{"id": "loop_01"}], "csv_header": "DATE,TIME,SENSOR_ID,LAT,LON,LANE,CLASS,SPEED_KMH,OCCUPANCY_MS"}
        except json.JSONDecodeError as e:
            logger.error(f"[LoopGenerator] Base schema configuration file is corrupted: {e}")
            return {"sensors": [{"id": "loop_01"}], "event_template": {}, "packet_template": {}}

    def generate(self, ground_truth: Dict[str, Any], timestamp: datetime.datetime) -> None:
        """Generates CSV counting files for all configured loops."""
        
        # MAESTRO MACRO-GAP CHECK: Abort generation if the physics engine injected a blackout
        if ground_truth.get('vehicle_flow') is None or ground_truth.get('current_speed') is None:
            return

        for idx, sensor_id in enumerate(self.loop_ids):
            if self.problems['gaps'] and random.random() < DEFAULT_GAP_PROB_LOOP:
                continue

            loop_loc = self.loop_locations[idx] if idx < len(self.loop_locations) else {"lat": DEFAULT_FALLBACK_LAT, "lon": DEFAULT_FALLBACK_LON}

            local_volume_factor: float = random.uniform(LOOP_VOLUME_FACTOR_MIN, LOOP_VOLUME_FACTOR_MAX)
            # Garantir pelo menos 1 veículo (mesmo de madrugada) para o arquivo não ficar vazio
            vehicle_count: int = max(1, int(ground_truth['vehicle_flow'] * local_volume_factor))
            
            # CSV Header
            csv_content: List[str] = [self.base_data.get("csv_header", "DATE,TIME,SENSOR_ID,LAT,LON,LANE,CLASS,SPEED_KMH,OCCUPANCY_MS")]
            
            for _ in range(vehicle_count):
                offset: int = random.randint(0, 5)
                event_time: datetime.datetime = timestamp + datetime.timedelta(seconds=offset)
                
                date_str: str = event_time.strftime("%Y-%m-%d")
                time_str: str = event_time.strftime("%H:%M:%S")
                
                # Lane Logic: 1 (Fast/Light) vs 2 (Slow/Heavy)
                lane: int = random.choices([1, 2], weights=LOOP_LANE_WEIGHTS)[0]
                
                if lane == 1:
                    v_class: str = random.choice(["LIGHT", "LIGHT", "MOTORCYCLE"])
                    speed: int = int(ground_truth['current_speed'] * random.uniform(LOOP_LANE_1_SPEED_MULTIPLIER_MIN, LOOP_LANE_1_SPEED_MULTIPLIER_MAX))
                else:
                    v_class = random.choice(["LIGHT", "HEAVY", "HEAVY", "BUS"])
                    speed = int(ground_truth['current_speed'] * random.uniform(LOOP_LANE_2_SPEED_MULTIPLIER_MIN, LOOP_LANE_2_SPEED_MULTIPLIER_MAX))

                # Physics for Occupancy Time
                length: float = LOOP_LIGHT_VEHICLE_LENGTH if v_class in ["LIGHT", "MOTORCYCLE"] else LOOP_HEAVY_VEHICLE_LENGTH
                speed_ms: float = max(1.0, speed / 3.6)
                occupancy: int = int((length / speed_ms) * 1000)
                
                lat: float = loop_loc.get("lat", DEFAULT_FALLBACK_LAT)
                lon: float = loop_loc.get("lon", DEFAULT_FALLBACK_LON)
                line: str = f"{date_str},{time_str},{sensor_id},{lat},{lon},{lane},{v_class},{speed},{occupancy}"
                csv_content.append(line)

            if self.problems['anomalies'] and random.random() < DEFAULT_ANOMALY_PROB_LOOP:
                csv_content = ["#ERROR: CONTROLLER RESTART", "#DUMP_FAILED"]

            # Save to specific folder: output/loop/loop_XX/file.csv
            ts_string: str = timestamp.strftime("%Y%m%d_%H%M%S")
            filename: str = f"{sensor_id}_{ts_string}.csv"
            sensor_dir: str = os.path.join(self.output_dir, sensor_id)
            filepath: str = os.path.join(sensor_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("\n".join(csv_content))
            except (FileNotFoundError, PermissionError, OSError) as e:
                logger.error(f"LoopGenerator IO Error on {sensor_id}: {e}")