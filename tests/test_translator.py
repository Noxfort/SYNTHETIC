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
# File: tests/test_translator.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import unittest
from ui.translator import translator

class TestTranslator(unittest.TestCase):
    def test_default_locale(self):
        self.assertEqual(translator.get_locale(), "en")

    def test_set_locale_valid(self):
        translator.set_locale("pt-br")
        self.assertEqual(translator.get_locale(), "pt-br")
        # Reset back to "en"
        translator.set_locale("en")
        self.assertEqual(translator.get_locale(), "en")

    def test_set_locale_invalid(self):
        translator.set_locale("invalid-locale")
        # Should remain the last set valid locale
        self.assertEqual(translator.get_locale(), "en")

    def test_translation_key_fallback(self):
        # Non-existent key should return key in brackets
        self.assertEqual(translator.t("non_existent_key_xyz"), "[non_existent_key_xyz]")

    def test_translation_formatting(self):
        # We test translation with arguments (fallback returns key in brackets or formatted text if formatting works)
        # Standard behaviour is to return fallback when not found
        self.assertEqual(translator.t("non_existent_key_xyz", "arg1"), "[non_existent_key_xyz]")

if __name__ == "__main__":
    unittest.main()
