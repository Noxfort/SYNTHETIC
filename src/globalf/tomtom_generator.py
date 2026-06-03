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
# File: globalf/tomtom_generator.py
# Author: Gabriel Moraes
# Date: 2026-02-27

import json
import os
import datetime
import random
import copy
import math
from ui.interfaces import IDataGenerator



class TomTomGenerator(IDataGenerator):
    """
    Generates exclusively TomTom data.
    Reads coordinates from external route_data.json file.
    Enforces a minimum generation interval of 5 minutes (300 seconds).
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = os.path.join(config['output_directory'], 'tomtom')
        self.problems = config['problems']
        self.last_generation_time = None
        
        os.makedirs(self.output_dir, exist_ok=True)

        # Load Coordinates from OSM Map Provider
        self.coordinates = self._extract_coordinates_from_map()
        
        # Load Base Model
        self.base_model = self._load_base_model()

        # Calculate distance
        self.segment_length_meters = self._calculate_route_length(self.coordinates)
        print(f"[TomTomGenerator] Route distance: {self.segment_length_meters:.2f} meters")

    def _load_base_model(self) -> dict:
        """Loads base structure from src/base/tomtom.json."""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_path, 'base', 'tomtom.json')
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data.get('base_flow', {})
        except Exception as e:
            print(f"[CRITICAL] Could not load tomtom.json: {e}")
            return {}

    def _extract_coordinates_from_map(self) -> list:
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
            return coords
        return [{'latitude': -23.3, 'longitude': -51.1}] # Fallback

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates distance in meters between two points."""
        R = 6371000 # Earth radius
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0)**2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda / 2.0)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _calculate_route_length(self, coordinates: list) -> float:
        """Sums the distance of all points in the coordinate list."""
        if not coordinates: return 0.0
        total_distance = 0.0
        for i in range(len(coordinates) - 1):
            p1 = coordinates[i]
            p2 = coordinates[i+1]
            dist = self._haversine_distance(p1['latitude'], p1['longitude'], 
                                            p2['latitude'], p2['longitude'])
            total_distance += dist
        return total_distance

    def generate(self, ground_truth: dict, timestamp: datetime.datetime) -> None:
        """Generates the JSON file for TomTom."""
        
        # MAESTRO MACRO-GAP CHECK: Abort generation if the physics engine injected a blackout
        if ground_truth.get('vehicle_flow') is None or ground_truth.get('current_speed') is None:
            return
            
        # Enforce 5-minute (300 seconds) minimum interval
        if self.last_generation_time is not None:
            time_since_last = (timestamp - self.last_generation_time).total_seconds()
            if time_since_last < 300:
                return
                
        self.last_generation_time = timestamp

        if self.problems['gaps'] and random.random() < 0.1:
            return

        data = copy.deepcopy(self.base_model)
        
        # Inject coordinates
        data['flowSegmentData']['coordinates'] = self.coordinates

        current_speed_kmh = float(ground_truth['current_speed'])
        free_flow_speed_kmh = float(ground_truth['free_flow_speed'])
        
        curr_speed_ms = max(0.1, current_speed_kmh / 3.6)
        free_speed_ms = max(0.1, free_flow_speed_kmh / 3.6)

        current_travel_time = int(self.segment_length_meters / curr_speed_ms)
        free_flow_travel_time = int(self.segment_length_meters / free_speed_ms)

        seg = data['flowSegmentData']
        seg['currentSpeed'] = int(current_speed_kmh)
        seg['freeFlowSpeed'] = int(free_flow_speed_kmh)
        seg['currentTravelTime'] = current_travel_time
        seg['freeFlowTravelTime'] = free_flow_travel_time

        if self.problems['anomalies'] and random.random() < 0.05:
            seg['currentSpeed'] = random.choice([-10, 999])

        ts_string = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"tomtom_flow_{ts_string}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"TomTomGenerator Error: {e}")