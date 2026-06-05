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
# File: core/dependency_checker.py
# Author: Gabriel Moraes
# Date: 2026-06-03

import importlib
import sys
from typing import Dict, Tuple
from src.core.logger import logger

class DependencyError(ImportError):
    """Custom exception raised when a critical heavy dependency is missing or corrupted."""
    pass

class DependencyChecker:
    """
    Validates heavy libraries and critical dependencies at system boot.
    Provides actionable installation instructions instead of generic tracebacks.
    """

    DEPENDENCIES: Dict[str, str] = {
        "torch": "pip install torch",
        "optuna": "pip install optuna",
        "llama_cpp": "pip install llama-cpp-python",
        "tkintermapview": "pip install tkintermapview",
        "haversine": "pip install haversine",
        "torch_geometric": "pip install torch-geometric",
        "numpy": "pip install numpy",
        "lxml": "pip install lxml"
    }

    @classmethod
    def check_dependency(cls, module_name: str) -> Tuple[bool, str]:
        """
        Validates if a single dependency is importable and returns its version.
        """
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            return True, f"Found {module_name} (v{version})"
        except ImportError as e:
            install_cmd = cls.DEPENDENCIES.get(module_name, f"pip install {module_name}")
            return False, f"Missing dependency '{module_name}'. Install using: {install_cmd} ({str(e)})"

    @classmethod
    def verify_all_dependencies(cls, fail_on_missing: bool = True) -> bool:
        """
        Verifies all listed dependencies.
        If fail_on_missing is True, raises DependencyError on any failure.
        """
        logger.info("Initializing Heavy Dependency Check...")
        missing_count = 0
        
        for dep in cls.DEPENDENCIES:
            success, msg = cls.check_dependency(dep)
            if success:
                logger.info(f"[ OK ] {msg}")
            else:
                logger.error(f"[FAIL] {dep} verification failed: {msg}")
                missing_count += 1

        if missing_count > 0:
            err_msg = f"Dependency check failed with {missing_count} missing package(s). Please fix environment dependencies."
            if fail_on_missing:
                raise DependencyError(err_msg)
            return False

        logger.info("All system dependencies validated successfully.")
        return True
