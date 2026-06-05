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
# File: tests/test_generators.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import unittest
import os
import shutil
import tempfile
import datetime
from src.globalf.waze_generator import WazeGenerator
from src.globalf.tomtom_generator import TomTomGenerator
from src.localf.camera_generator import CameraGenerator
from src.localf.loop_generator import LoopGenerator
from src.core.map_provider import OSMMapProvider

class TestDataGenerators(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output
        self.test_dir = tempfile.mkdtemp()
        
        # Mock map provider
        self.mock_map = OSMMapProvider()
        self.mock_map.nodes = {
            "node_1": (-23.3, -51.1),
            "node_2": (-23.301, -51.101)
        }
        self.mock_map.ways = [["node_1", "node_2"]]
        self.mock_map.bounds = (-23.31, -51.11, -23.29, -51.09)

        self.config = {
            "output_directory": self.test_dir,
            "problems": {
                "gaps": False,
                "anomalies": False
            },
            "simulation": {
                "num_cameras": 2,
                "num_loops": 2
            },
            "map": {
                "provider": self.mock_map,
                "local_points": {
                    "cameras": [{"lat": -23.3, "lon": -51.1}],
                    "loops": [{"lat": -23.3, "lon": -51.1}]
                }
            }
        }
        self.ground_truth = {
            "vehicle_flow": 100,
            "current_speed": 60,
            "free_flow_speed": 70
        }
        self.timestamp = datetime.datetime(2026, 6, 1, 8, 0, 0)

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_waze_generator(self):
        generator = WazeGenerator(self.config)
        generator.generate(self.ground_truth, self.timestamp)
        
        # Verify files are created
        waze_dir = os.path.join(self.test_dir, "waze")
        self.assertTrue(os.path.exists(waze_dir))
        files = os.listdir(waze_dir)
        self.assertTrue(len(files) > 0)

    def test_tomtom_generator(self):
        generator = TomTomGenerator(self.config)
        generator.generate(self.ground_truth, self.timestamp)
        
        # Verify files are created
        tomtom_dir = os.path.join(self.test_dir, "tomtom")
        self.assertTrue(os.path.exists(tomtom_dir))
        files = os.listdir(tomtom_dir)
        self.assertTrue(len(files) > 0)

    def test_camera_generator(self):
        generator = CameraGenerator(self.config)
        generator.generate(self.ground_truth, self.timestamp)
        
        # Verify files are created
        camera_dir = os.path.join(self.test_dir, "camera")
        self.assertTrue(os.path.exists(camera_dir))
        self.assertTrue(os.path.exists(os.path.join(camera_dir, "cam_01")))
        self.assertTrue(os.path.exists(os.path.join(camera_dir, "cam_02")))

    def test_loop_generator(self):
        generator = LoopGenerator(self.config)
        generator.generate(self.ground_truth, self.timestamp)
        
        # Verify files are created
        loop_dir = os.path.join(self.test_dir, "loop")
        self.assertTrue(os.path.exists(loop_dir))
        self.assertTrue(os.path.exists(os.path.join(loop_dir, "loop_01")))
        self.assertTrue(os.path.exists(os.path.join(loop_dir, "loop_02")))

if __name__ == "__main__":
    unittest.main()
