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
from ui.interfaces import IDataGenerator

class CameraGenerator(IDataGenerator):
    """
    Generates exclusively Camera (LPR/OCR) data in JSON format.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = os.path.join(config['output_directory'], 'camera')
        self.problems = config['problems']
        self.sim_config = config['simulation']
        self.camera_locations = config.get('map', {}).get('local_points', {}).get('cameras', [])
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load base from camera.json
        self.base_data = self._load_base()
        
        # Quantity configuration
        num_cameras = self.sim_config.get('num_cameras', 1)
        
        # ID GENERATION
        self.camera_ids = [f"cam_{i:02d}" for i in range(1, num_cameras + 1)]

        # Pre-creation of individual folders
        for cam_id in self.camera_ids:
            os.makedirs(os.path.join(self.output_dir, cam_id), exist_ok=True)

    def _load_base(self) -> dict:
        """Loads base structure from src/base/camera.json."""
        import json
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_path, 'src', 'base', 'camera.json')

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[CRITICAL] Could not load camera.json: {e}")
            return {"sensors": [{"id": "cam_01"}], "event_template": {}, "packet_template": {}}

    def _generate_plate(self):
        """Generates a random license plate (Mercosul or Old format)."""
        if random.random() > 0.5:
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

    def _uuid_short(self):
        """Helper to generate a short UUID string."""
        return str(random.randint(100000, 999999))

    def generate(self, ground_truth: dict, timestamp: datetime.datetime) -> None:
        """Generates LPR JSON files for all configured cameras."""
        
        # MAESTRO MACRO-GAP CHECK: Abort generation if the physics engine injected a blackout
        if ground_truth.get('vehicle_flow') is None or ground_truth.get('current_speed') is None:
            return

        for idx, sensor_id in enumerate(self.camera_ids):
            if self.problems['gaps'] and random.random() < 0.15:
                continue

            cam_loc = self.camera_locations[idx] if idx < len(self.camera_locations) else {"lat": 0.0, "lon": 0.0}

            local_count_factor = random.uniform(0.08, 0.12)
            # Garantir pelo menos 1 veículo (mesmo de madrugada) para o arquivo não ser pulado
            vehicle_count = max(1, int(ground_truth['vehicle_flow'] * local_count_factor))

            events = []
            
            for _ in range(vehicle_count):
                plate_text = self._generate_plate()
                confidence = round(random.uniform(85.0, 99.9), 1)
                
                # Camera speed slightly noisy compared to ground truth
                # ground_truth speed is in km/h
                current_speed = ground_truth['current_speed']
                speed_read = current_speed * random.uniform(0.9, 1.1)
                import copy
                event = copy.deepcopy(self.base_data.get("event_template", {}))
                event["timestamp"] = timestamp.isoformat() + "Z"
                event["camera_id"] = sensor_id
                event["uuid"] = f"{sensor_id}-{self._uuid_short()}"
                
                # Make sure nested dicts exist
                if "location" not in event: event["location"] = {}
                event["location"]["lat"] = cam_loc["lat"]
                event["location"]["lon"] = cam_loc["lon"]
                # Align with camera.json template
                event["vehicle"]["plate_string"] = plate_text
                event["vehicle"]["confidence"] = confidence
                event["vehicle"]["attributes"]["color"] = random.choice(["white", "silver", "black", "red", "gray"])
                event["vehicle"]["attributes"]["type"] = random.choice(["car", "car", "suv", "truck", "motorcycle"])
                event["vehicle"]["attributes"]["estimated_speed_kmh"] = round(speed_read, 1)
                
                if "images" not in event: event["images"] = {}
                event["images"]["snapshot_url"] = f"http://10.0.0.5/{sensor_id}/{timestamp.strftime('%Y%m%d')}/{plate_text}.jpg"
                
                events.append(event)

            output_data = copy.deepcopy(self.base_data.get("packet_template", {}))
            output_data["packet_id"] = self._uuid_short()
            output_data["sensor_id"] = sensor_id
            output_data["timestamp_sent"] = datetime.datetime.now().isoformat()
            output_data["recognitions"] = events

            if self.problems['anomalies'] and random.random() < 0.05:
                output_data["recognitions"] = "ERROR_BUFFER_OVERFLOW"

            ts_string = timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{sensor_id}_{ts_string}.json"
            sensor_dir = os.path.join(self.output_dir, sensor_id)
            filepath = os.path.join(sensor_dir, filename)
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(output_data, f, indent=2)
            except Exception as e:
                print(f"CameraGenerator Error {sensor_id}: {e}")