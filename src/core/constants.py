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
# File: core/constants.py
# Author: Gabriel Moraes
# Date: 2026-06-03

from typing import List

# --- Geographic & Physical Constants ---
EARTH_RADIUS_METERS: float = 6371000.0
KMH_TO_MS_MULTIPLIER: float = 3.6
MS_TO_KMH_MULTIPLIER: float = 3.6

# --- Londrina Bounding Box / Fallback Coordinates ---
DEFAULT_FALLBACK_LAT: float = -23.3
DEFAULT_FALLBACK_LON: float = -51.1

# --- Generation Constraints ---
MIN_GENERATION_INTERVAL_SECS: int = 300
DEFAULT_SPEED_CLAMP_MIN: int = 20
DEFAULT_SPEED_CLAMP_MAX: int = 110

# --- Temporal Inertia (Director) ---
TEMPORAL_INERTIA_ALPHA: float = 0.7  # 70% new day scenario, 30% previous day

# --- Sensor Gap & Anomaly Probabilities ---
DEFAULT_GAP_PROB_TOMTOM: float = 0.1
DEFAULT_GAP_PROB_WAZE: float = 0.1
DEFAULT_GAP_PROB_CAMERA: float = 0.15
DEFAULT_GAP_PROB_LOOP: float = 0.10

DEFAULT_ANOMALY_PROB_TOMTOM: float = 0.05
DEFAULT_ANOMALY_PROB_CAMERA: float = 0.05
DEFAULT_ANOMALY_PROB_LOOP: float = 0.03
DEFAULT_ANOMALY_PROB_ORCHESTRATOR_GAPS: float = 0.03
DEFAULT_ANOMALY_PROB_ORCHESTRATOR_ANOMALIES: float = 0.05

# --- Baseline Math for scaling CSDI output (Physics Layer) ---
CSDI_BASE_FLOW: int = 500
CSDI_FLOW_VARIATION: int = 300
CSDI_BASE_SPEED: int = 60
CSDI_SPEED_VARIATION: int = 25

# --- Camera Sensor Attributes ---
CAMERA_LANE_VOLUME_FACTOR_MIN: float = 0.08
CAMERA_LANE_VOLUME_FACTOR_MAX: float = 0.12
CAMERA_OCR_CONFIDENCE_MIN: float = 85.0
CAMERA_OCR_CONFIDENCE_MAX: float = 99.9
CAMERA_PLATE_FORMAT_PROB: float = 0.5  # 50% Mercosul, 50% Old format
CAMERA_SPEED_NOISE_MIN: float = 0.9
CAMERA_SPEED_NOISE_MAX: float = 1.1

# --- Loop Sensor Attributes ---
LOOP_VOLUME_FACTOR_MIN: float = 0.04
LOOP_VOLUME_FACTOR_MAX: float = 0.06
LOOP_LANE_WEIGHTS: List[float] = [0.6, 0.4]
LOOP_LANE_1_SPEED_MULTIPLIER_MIN: float = 0.95
LOOP_LANE_1_SPEED_MULTIPLIER_MAX: float = 1.20
LOOP_LANE_2_SPEED_MULTIPLIER_MIN: float = 0.70
LOOP_LANE_2_SPEED_MULTIPLIER_MAX: float = 0.95
LOOP_LIGHT_VEHICLE_LENGTH: float = 4.0
LOOP_HEAVY_VEHICLE_LENGTH: float = 12.0
