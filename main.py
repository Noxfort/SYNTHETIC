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
# File: main.py
# Author: Gabriel Moraes
# Date: 2025-11-27

import threading
import sys
import os

def run_simulation(config, on_success, on_error):
    """
    Executed in the background thread.
    Imports core logic and runs the simulation orchestrator.
    """
    try:
        from src.core.simulation_logic import SimulationOrchestrator
        from src.globalf.tomtom_generator import TomTomGenerator
        from src.globalf.waze_generator import WazeGenerator
        from src.localf.camera_generator import CameraGenerator
        from src.localf.loop_generator import LoopGenerator
        
        active_generators = []
        if config['sources']['tomtom']: active_generators.append(TomTomGenerator(config))
        if config['sources']['waze']: active_generators.append(WazeGenerator(config))
        if config['sources']['camera']: active_generators.append(CameraGenerator(config))
        if config['sources']['loop']: active_generators.append(LoopGenerator(config))

        orchestrator = SimulationOrchestrator(config, active_generators)
        orchestrator.run()
        
        on_success(config["output_directory"])
    
    except Exception as e:
        on_error(e)

def start_simulation(config, on_success, on_error):
    """
    Called by the GUI to start the simulation process.
    Spawns a thread so the UI is not blocked.
    """
    generation_thread = threading.Thread(
        target=run_simulation,
        args=(config, on_success, on_error),
        daemon=True
    )
    generation_thread.start()

if __name__ == "__main__":
    # Ensure the script runs with the current directory in PYTHONPATH
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from ui.gui import DataGeneratorApp
    
    # Initialize UI and inject the start_simulation callback
    app = DataGeneratorApp(on_generate_callback=start_simulation)
    app.mainloop()