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
from typing import Dict, Any, Callable
import torch

# Ativar TF32 globalmente para GPUs modernas (Ampere+)
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        pass

from src.core.logger import logger
from src.core.dependency_checker import DependencyChecker, DependencyError

def run_simulation(config: Dict[str, Any], on_success: Callable[[str], None], on_error: Callable[[Exception], None]) -> None:
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
    
    except (ImportError, ValueError, RuntimeError) as e:
        logger.error(f"[main] Simulation failed during setup/execution: {e}", exc_info=True)
        on_error(e)
    except Exception as e:
        logger.critical(f"[main] Unexpected catastrophic error in simulation: {e}", exc_info=True)
        on_error(e)

def start_simulation(config: Dict[str, Any], on_success: Callable[[str], None], on_error: Callable[[Exception], None]) -> None:
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
    
    # Run heavy dependency validation before GUI boot
    try:
        DependencyChecker.verify_all_dependencies(fail_on_missing=True)
    except DependencyError as err:
        logger.critical(f"FATAL: Dependency check failed: {err}")
        sys.exit(1)
        
    from ui.gui import DataGeneratorApp
    
    # Initialize UI and inject the start_simulation callback
    app = DataGeneratorApp(on_generate_callback=start_simulation)
    app.mainloop()