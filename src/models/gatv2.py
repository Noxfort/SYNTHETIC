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
# File: models/gatv2.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
except ImportError:
    # Fallback or stub if torch_geometric is not installed in the environment yet
    GATv2Conv = None
    global_mean_pool = None

class LightweightGATv2(nn.Module):
    """
    Lightweight Graph Attention Network v2 for mapping topology.
    Converts road network nodes and edges into a spatial context embedding.
    """
    def __init__(self, in_channels=2, hidden_channels=16, out_channels=32, heads=4):
        super(LightweightGATv2, self).__init__()
        
        if GATv2Conv is None:
            raise ImportError("torch_geometric is required for GATv2. Please install it.")
            
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        # On the second layer, we want the output to be exactly out_channels, so we concat=False
        # or we just let it output out_channels directly
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        
    def forward(self, x, edge_index, batch=None):
        """
        x: Node features [num_nodes, in_channels] (e.g., lat, lon)
        edge_index: Graph connectivity [2, num_edges]
        batch: Batch vector assigning each node to a graph (optional, defaults to all 0s)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # GATv2 Convolution 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # GATv2 Convolution 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Global Pooling to get a single embedding vector for the entire graph
        graph_embedding = global_mean_pool(x, batch) # Shape: [num_graphs, out_channels]
        
        return graph_embedding

    def extract_context(self, map_provider):
        """
        Helper method to extract context directly from the OSMMapProvider.
        """
        import numpy as np
        node_features, edge_index = map_provider.parse_osm_to_graph()
        
        if len(node_features) == 0:
            # Fallback if map is empty
            return torch.zeros((1, 32))
            
        x_tensor = torch.tensor(node_features, dtype=torch.float32)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        
        # Disable gradient computation for pure feature extraction
        with torch.no_grad():
            self.eval()
            embedding = self.forward(x_tensor, edge_index_tensor)
            
        return embedding
