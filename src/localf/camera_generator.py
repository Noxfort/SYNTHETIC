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
# File: localf/camera_generator.py
# Author: Gabriel Moraes
# Date: 2026-02-27

import json
import os
import datetime
import random
import string
import copy
from typing import Dict, List, Any, Optional
from ui.interfaces import IDataGenerator
from src.core.logger import logger
from src.core.constants import (
    DEFAULT_FALLBACK_LAT,
    DEFAULT_FALLBACK_LON,
    DEFAULT_GAP_PROB_CAMERA,
    DEFAULT_ANOMALY_PROB_CAMERA,
    CAMERA_LANE_VOLUME_FACTOR_MIN,
    CAMERA_LANE_VOLUME_FACTOR_MAX,
    CAMERA_OCR_CONFIDENCE_MIN,
    CAMERA_OCR_CONFIDENCE_MAX,
    CAMERA_PLATE_FORMAT_PROB,
    CAMERA_SPEED_NOISE_MIN,
    CAMERA_SPEED_NOISE_MAX
)

class CameraGenerator(IDataGenerator):
    """
    Generates exclusively Camera (LPR/OCR) data in JSON format.
    """
    
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.output_dir: str = os.path.join(config['output_directory'], 'camera')
        self.problems: dict = config['problems']
        self.sim_config: dict = config['simulation']
        self.camera_locations: List[Dict[str, float]] = config.get('map', {}).get('local_points', {}).get('cameras', [])
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load base from camera.json
        self.base_data: dict = self._load_base()
        
        # Quantity configuration
        num_cameras: int = self.sim_config.get('num_cameras', 1)
        
        # ID GENERATION
        self.camera_ids: List[str] = [f"cam_{i:02d}" for i in range(1, num_cameras + 1)]

        # Pre-creation of individual folders
        for cam_id in self.camera_ids:
            os.makedirs(os.path.join(self.output_dir, cam_id), exist_ok=True)

    def _load_base(self) -> dict:
        """Loads base structure from src/base/camera.json."""
        base_path: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path: str = os.path.join(base_path, 'src', 'base', 'camera.json')

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"[CameraGenerator] Base schema config not found: {e}")
            return {"sensors": [{"id": "cam_01"}], "event_template": {}, "packet_template": {}}
        except json.JSONDecodeError as e:
            logger.error(f"[CameraGenerator] Base schema configuration file is corrupted: {e}")
            return {"sensors": [{"id": "cam_01"}], "event_template": {}, "packet_template": {}}

    def _generate_plate(self) -> str:
        """Generates a random license plate (Mercosul or Old format)."""
        if random.random() > CAMERA_PLATE_FORMAT_PROB:
            # Mercosul: ABC1D23
            L1 = ''.join(random.choices(string.ascii_uppercase, k=3))
            N1 = random.choice(string.digits)
            L2 = random.choice(string.ascii_uppercase)
            N2 = ''.join(random.choices(string.digits, k=2))
            return f"{L1}{N1}{L2}{N2}"
        else:
            # Old: ABC1234
            L = ''.join(random.choices(string.ascii_uppercase, k=3))
            N = ''.join(random.choices(string.digits, k=4))
            return f"{L}{N}"

    def _uuid_short(self) -> str:
        """Helper to generate a short UUID string."""
        return str(random.randint(100000, 999999))

    def generate(self, ground_truth: Dict[str, Any], timestamp: datetime.datetime) -> None:
        """Generates LPR JSON files for all configured cameras."""
        
        # MAESTRO MACRO-GAP CHECK: Abort generation if the physics engine injected a blackout
        if ground_truth.get('vehicle_flow') is None or ground_truth.get('current_speed') is None:
            return

        for idx, sensor_id in enumerate(self.camera_ids):
            if self.problems['gaps'] and random.random() < DEFAULT_GAP_PROB_CAMERA:
                continue

            cam_loc = self.camera_locations[idx] if idx < len(self.camera_locations) else {"lat": DEFAULT_FALLBACK_LAT, "lon": DEFAULT_FALLBACK_LON}

            local_count_factor: float = random.uniform(CAMERA_LANE_VOLUME_FACTOR_MIN, CAMERA_LANE_VOLUME_FACTOR_MAX)
            # Garantir pelo menos 1 veículo (mesmo de madrugada) para o arquivo não ser pulado
            vehicle_count: int = max(1, int(ground_truth['vehicle_flow'] * local_count_factor))

            events: List[dict] = []
            
            for _ in range(vehicle_count):
                plate_text: str = self._generate_plate()
                confidence: float = round(random.uniform(CAMERA_OCR_CONFIDENCE_MIN, CAMERA_OCR_CONFIDENCE_MAX), 1)
                
                # Camera speed slightly noisy compared to ground truth
                # ground_truth speed is in km/h
                current_speed: float = float(ground_truth['current_speed'])
                speed_read: float = current_speed * random.uniform(CAMERA_SPEED_NOISE_MIN, CAMERA_SPEED_NOISE_MAX)
                
                event: dict = copy.deepcopy(self.base_data.get("event_template", {}))
                event["timestamp"] = timestamp.isoformat() + "Z"
                event["camera_id"] = sensor_id
                event["uuid"] = f"{sensor_id}-{self._uuid_short()}"
                
                # Make sure nested dicts exist
                if "location" not in event: 
                    event["location"] = {}
                event["location"]["lat"] = cam_loc.get("lat", DEFAULT_FALLBACK_LAT)
                event["location"]["lon"] = cam_loc.get("lon", DEFAULT_FALLBACK_LON)
                
                if "vehicle" not in event:
                    event["vehicle"] = {"attributes": {}}
                event["vehicle"]["plate_string"] = plate_text
                event["vehicle"]["confidence"] = confidence
                event["vehicle"]["attributes"]["color"] = random.choice(["white", "silver", "black", "red", "gray"])
                event["vehicle"]["attributes"]["type"] = random.choice(["car", "car", "suv", "truck", "motorcycle"])
                event["vehicle"]["attributes"]["estimated_speed_kmh"] = round(speed_read, 1)
                
                if "images" not in event: 
                    event["images"] = {}
                event["images"]["snapshot_url"] = f"http://10.0.0.5/{sensor_id}/{timestamp.strftime('%Y%m%d')}/{plate_text}.jpg"
                
                events.append(event)

            output_data: dict = copy.deepcopy(self.base_data.get("packet_template", {}))
            output_data["packet_id"] = self._uuid_short()
            output_data["sensor_id"] = sensor_id
            output_data["timestamp_sent"] = datetime.datetime.now().isoformat()
            output_data["recognitions"] = events

            if self.problems['anomalies'] and random.random() < DEFAULT_ANOMALY_PROB_CAMERA:
                output_data["recognitions"] = "ERROR_BUFFER_OVERFLOW"

            ts_string: str = timestamp.strftime("%Y%m%d_%H%M%S")
            filename: str = f"{sensor_id}_{ts_string}.json"
            sensor_dir: str = os.path.join(self.output_dir, sensor_id)
            filepath: str = os.path.join(sensor_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2)
            except (FileNotFoundError, PermissionError, OSError) as e:
                logger.error(f"CameraGenerator IO Error on camera {sensor_id}: {e}")