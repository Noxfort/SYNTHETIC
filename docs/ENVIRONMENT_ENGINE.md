# Environment Engine: Temporal & Meteorological Physics

The `SYNTHETIC` architecture isolates all physical laws of time and weather into the `EnvironmentManager` (`src/core/environment.py`), adhering to the Single Responsibility Principle (SRP). 

## 1. Temporal Ground Zero
To ensure datasets are cyclically coherent for ML training, simulations cannot start at random arbitrary dates. 
- The `get_next_monday_midnight()` function mathematically calculates the immediate upcoming Monday at `00:00:00`.
- This ensures every 7-day simulation cycle perfectly captures the ramp-up of a work week, the peak congestion of Wednesday/Thursday, and the decay into the weekend, standardizing the time-series output for the diffusion models.

## 2. Markov-Chain Weather Physics
Real-world weather does not shift randomly from a "Thunderstorm" at 2 PM to "Sunny" at 3 PM without passing through transitional states. The legacy random-choice system was decommissioned in favor of a Markov-chain state machine.

### Data-Driven Transitions (OCP)
The transitions are entirely separated from the Python logic into `config/weather_rules.json`. This respects the Open/Closed Principle, allowing meteorologists or engineers to tweak transition probabilities without touching the codebase.

**Example of Valid Transitions:**
- `Sunny` → `Partly Cloudy` or `Hazy`
- `Overcast` → `Rain`, `Drizzle`, or `Snow`
- `Heavy Rain` → `Thunderstorm`

### Triple-Tuple Representation
When requested by the `SimulationOrchestrator`, the `EnvironmentManager` outputs a weather state represented by three components:
1. **Condition**: The base state (e.g., "Rain").
2. **Intensity**: Based on a category map, ensures rain is "moderate" or "heavy", not "settled".
3. **Characteristic**: Appends narrative physical implications (e.g., "with slick roads").

These are joined into a cohesive string (e.g., *"Moderate rain with slick roads"*) and injected directly into the Phi-4 SLM prompt. The SLM's advanced reasoning engine mathematically correlates "slick roads" with a necessary drop in physical free-flow speed, which is later enforced by the VAE-TCN Guardian.
