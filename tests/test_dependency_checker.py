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
# File: tests/test_dependency_checker.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import unittest
from src.core.dependency_checker import DependencyChecker, DependencyError

class TestDependencyChecker(unittest.TestCase):
    def test_check_dependency_valid(self):
        success, msg = DependencyChecker.check_dependency("numpy")
        self.assertTrue(success)
        self.assertIn("numpy", msg)

    def test_check_dependency_invalid(self):
        success, msg = DependencyChecker.check_dependency("non_existent_module_xyz")
        self.assertFalse(success)
        self.assertIn("non_existent_module_xyz", msg)

    def test_verify_all_dependencies_no_fail(self):
        # We check without raising an error
        try:
            res = DependencyChecker.verify_all_dependencies(fail_on_missing=False)
            self.assertIsInstance(res, bool)
        except DependencyError:
            self.fail("verify_all_dependencies raised DependencyError even with fail_on_missing=False")

if __name__ == "__main__":
    unittest.main()
