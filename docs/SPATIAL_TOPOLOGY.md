# Spatial Topology & OSM Map Integration

In the modern `SYNTHETIC` architecture, static and deterministic coordinate generation (`route_data.json`) has been entirely decommissioned. The engine now operates with full spatial awareness, anchoring simulated traffic data to real-world geographies via OpenStreetMap (`.osm`).

## 1. OSM Parsing (`OSMMapProvider`)

The `OSMMapProvider` (`src/core/map_provider.py`) acts as the bridge between raw XML data and Python data structures.

### Workflow:
1. **Ingestion**: The user selects a `.osm` file representing a specific city or bounding box.
2. **Node Extraction**: The provider reads all `<node>` elements to build a dictionary of latitude/longitude points.
3. **Way Filtering**: The provider isolates `<way>` elements containing the `highway` tag (e.g., `primary`, `residential`, `motorway`), discarding buildings and pedestrian paths.
4. **Bounding Box**: Automatically calculates the spatial limits to center the UI map interface.

### The "Snap-to-Road" Mechanism
Real-world sensors (cameras, inductive loops) are rarely placed perfectly on intersection nodes. When a user clicks the UI map to place a sensor, the `OSMMapProvider` iterates through the valid road edges and snaps the raw user coordinate to the mathematically nearest valid point on the topological graph using Haversine distance calculations. This guarantees that all generated `lat`/`lon` values in the output CSVs and JSONs reside perfectly on the pavement.

## 2. Graph Neural Extraction (`LightweightGATv2`)

While the `OSMMapProvider` gives us coordinates, neural diffusion models (like CSDI) cannot natively understand "latitude" and "longitude". They require high-dimensional tensor representations. 

### Why GATv2?
Graph Attention Networks v2 (GATv2) are designed to process unstructured grid topologies. By passing our map data through `LightweightGATv2` (`src/models/gatv2.py`), the model assigns attention weights to intersections (Nodes) and roads (Edges). 
- A 5-way chaotic intersection receives high attention.
- A long, uninterrupted straight highway receives a different attention signature.

### The Output
The GATv2 outputs a **Spatial Context Tensor**. During Phase 2, the `CSDIEngine` concatenates this spatial tensor with the SLM's latent dream vector. 
- The SLM provides the *behavioral* constraints (e.g., "Heavy Rain, High Traffic").
- The GATv2 provides the *physical* constraints (e.g., "This road only has two connected adjacent nodes").

Together, this guarantees that synthesized congestion waves propagate naturally across the specific grid of the city loaded by the user.
