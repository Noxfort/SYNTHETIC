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
# File: globalf/waze_generator.py
# Author: Gabriel Moraes
# Date: 2026-02-27

import json
import os
import datetime
import random
import copy
import uuid
import math
from typing import Dict, List, Any, Optional
from ui.interfaces import IDataGenerator
from src.core.logger import logger
from src.core.constants import (
    EARTH_RADIUS_METERS,
    KMH_TO_MS_MULTIPLIER,
    DEFAULT_FALLBACK_LAT,
    DEFAULT_FALLBACK_LON,
    MIN_GENERATION_INTERVAL_SECS,
    DEFAULT_GAP_PROB_WAZE
)


class WazeGenerator(IDataGenerator):
    """
    Generates exclusively Waze Feed data.
    Reads coordinates from external route_data.json file.
    Reads static templates from waze_config.json.
    Enforces a minimum generation interval of 5 minutes (300 seconds).
    """
    
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.output_dir: str = os.path.join(config['output_directory'], 'waze')
        self.problems: dict = config['problems']
        self.last_generation_time: Optional[datetime.datetime] = None
        
        os.makedirs(self.output_dir, exist_ok=True)

        self.coordinates: List[Dict[str, float]] = self._extract_coordinates_from_map()
        self.waze_templates: dict = self._load_templates()
        self.segment_length_meters: float = self._calculate_route_length(self.coordinates)
        
        logger.info(f"[WazeGenerator] Route distance: {self.segment_length_meters:.2f} meters")

    def _load_templates(self) -> dict:
        """Loads static templates and base structure from src/base/waze.json."""
        base_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path: str = os.path.join(base_path, 'base', 'waze.json')

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"[WazeGenerator] Could not find base template file waze.json: {e}")
            return {
                "base_feed": {"jams": [], "endTimeMillis": 0, "startTimeMillis": 0, "startTime": "", "endTime": ""},
                "jam": {"street": "Unknown", "city": "Unknown", "country": "BR", "type": "NONE"}
            }
        except json.JSONDecodeError as e:
            logger.error(f"[WazeGenerator] Base template file waze.json is corrupted: {e}")
            return {
                "base_feed": {"jams": [], "endTimeMillis": 0, "startTimeMillis": 0, "startTime": "", "endTime": ""},
                "jam": {"street": "Unknown", "city": "Unknown", "country": "BR", "type": "NONE"}
            }

    def _extract_coordinates_from_map(self) -> List[Dict[str, float]]:
        """Extracts a random road segment from the OSM Map Provider to serve as the global incident line."""
        map_provider = self.config.get('map', {}).get('provider')
        if map_provider and map_provider.ways:
            # Pick a random way
            import random
            way_nodes = random.choice(map_provider.ways)
            coords = []
            for nd in way_nodes:
                if nd in map_provider.nodes:
                    lat, lon = map_provider.nodes[nd]
                    coords.append({'latitude': lat, 'longitude': lon})
            if coords:
                return coords
        return [{'latitude': DEFAULT_FALLBACK_LAT, 'longitude': DEFAULT_FALLBACK_LON}] # Fallback

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates distance in meters between two points using central constants."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0)**2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda / 2.0)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return EARTH_RADIUS_METERS * c

    def _calculate_route_length(self, coordinates: List[Dict[str, float]]) -> float:
        """Sums the distance of all points in the coordinate list."""
        if not coordinates: 
            return 0.0
        total_distance: float = 0.0
        for i in range(len(coordinates) - 1):
            p1 = coordinates[i]
            p2 = coordinates[i+1]
            dist = self._haversine_distance(p1['latitude'], p1['longitude'], 
                                            p2['latitude'], p2['longitude'])
            total_distance += dist
        return total_distance

    def generate(self, ground_truth: Dict[str, Any], timestamp: datetime.datetime) -> None:
        """Generates the JSON file for Waze."""
        
        # MAESTRO MACRO-GAP CHECK: Abort generation if the physics engine injected a blackout
        if ground_truth.get('vehicle_flow') is None or ground_truth.get('current_speed') is None:
            return
        
        if self.last_generation_time is not None:
            time_since_last = (timestamp - self.last_generation_time).total_seconds()
            if time_since_last < MIN_GENERATION_INTERVAL_SECS:
                return
                
        self.last_generation_time = timestamp

        if self.problems['gaps'] and random.random() < DEFAULT_GAP_PROB_WAZE:
            return

        data: dict = copy.deepcopy(self.waze_templates.get("base_feed", {}))
        
        now_ms: int = int(timestamp.timestamp() * 1000)
        data['startTimeMillis'] = now_ms
        data['endTimeMillis'] = now_ms + 60000
        data['startTime'] = timestamp.isoformat()
        data['endTime'] = (timestamp + datetime.timedelta(seconds=60)).isoformat()

        current_speed_kmh: float = float(ground_truth['current_speed'])
        free_flow_speed_kmh: float = float(ground_truth['free_flow_speed'])
        
        curr_speed_ms: float = max(0.1, current_speed_kmh / KMH_TO_MS_MULTIPLIER)
        free_speed_ms: float = max(0.1, free_flow_speed_kmh / KMH_TO_MS_MULTIPLIER)

        real_time_sec: float = self.segment_length_meters / curr_speed_ms
        ideal_time_sec: float = self.segment_length_meters / free_speed_ms
        delay_seconds: int = max(0, int(real_time_sec - ideal_time_sec))

        ratio: float = current_speed_kmh / free_flow_speed_kmh if free_flow_speed_kmh > 0 else 0.0
        level: int = 0
        if ratio < 0.9: level = 1
        if ratio < 0.7: level = 2
        if ratio < 0.5: level = 3
        if ratio < 0.3: level = 4
        if current_speed_kmh < 5: level = 5

        # Merge dynamic data with static template for JAM
        jam: dict = copy.deepcopy(self.waze_templates.get("jam", {}))
        jam["uuid"] = str(uuid.uuid4())
        jam["pubMillis"] = now_ms
        jam["speed"] = int(current_speed_kmh)
        jam["length"] = int(self.segment_length_meters)
        jam["level"] = level
        jam["delay"] = delay_seconds
        jam["line"] = [{"x": p['longitude'], "y": p['latitude']} for p in self.coordinates]
        
        if 'jams' not in data:
            data['jams'] = []
        data['jams'].append(jam)

        # Alert generation removed to simplify JSON output.

        ts_string: str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename: str = f"waze_feed_{ts_string}.json"
        filepath: str = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"WazeGenerator IO Error on {filename}: {e}")