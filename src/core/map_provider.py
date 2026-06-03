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
# File: core/map_provider.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import xml.etree.ElementTree as ET
from haversine import haversine
import numpy as np
import math

class OSMMapProvider:
    def __init__(self):
        self.nodes = {} # {id: (lat, lon)}
        self.ways = [] # list of lists of node IDs
        self.road_nodes = set()
        self.bounds = None # (min_lat, min_lon, max_lat, max_lon)
        
    def parse_osm_file(self, filepath):
        """Parses the OSM file to extract bounds, nodes, and highway ways."""
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Extract bounds
        bounds_tag = root.find('bounds')
        if bounds_tag is not None:
            self.bounds = (
                float(bounds_tag.attrib['minlat']),
                float(bounds_tag.attrib['minlon']),
                float(bounds_tag.attrib['maxlat']),
                float(bounds_tag.attrib['maxlon'])
            )
        
        # Extract all nodes
        min_lat, min_lon = float('inf'), float('inf')
        max_lat, max_lon = float('-inf'), float('-inf')
        
        for node in root.findall('node'):
            node_id = node.attrib['id']
            lat = float(node.attrib['lat'])
            lon = float(node.attrib['lon'])
            self.nodes[node_id] = (lat, lon)
            
            # If bounds tag is missing, we calculate it
            if self.bounds is None:
                min_lat = min(min_lat, lat)
                min_lon = min(min_lon, lon)
                max_lat = max(max_lat, lat)
                max_lon = max(max_lon, lon)
        
        if self.bounds is None and self.nodes:
            self.bounds = (min_lat, min_lon, max_lat, max_lon)
            
        # Extract ways (only highways)
        for way in root.findall('way'):
            is_highway = False
            for tag in way.findall('tag'):
                if tag.attrib['k'] == 'highway':
                    is_highway = True
                    break
            
            if is_highway:
                way_nodes = [nd.attrib['ref'] for nd in way.findall('nd')]
                self.ways.append(way_nodes)
                for nd in way_nodes:
                    self.road_nodes.add(nd)

    def get_bounds(self):
        return self.bounds
        
    def snap_to_road(self, lat, lon):
        """Finds the closest point on any road segment to the given coordinates."""
        if not self.ways:
            return lat, lon # Fallback
            
        min_dist = float('inf')
        best_point = (lat, lon)
        
        # Approximate local flat distance considering Earth curvature at this latitude
        lat_rad = math.radians(lat)
        lon_scale = math.cos(lat_rad)
        
        def dist_squared(p1, p2):
            dy = p1[0] - p2[0]
            dx = (p1[1] - p2[1]) * lon_scale
            return dy**2 + dx**2
            
        def closest_point_on_segment(p, a, b):
            dy_ab = b[0] - a[0]
            dx_ab = (b[1] - a[1]) * lon_scale
            ab_len_sq = dy_ab**2 + dx_ab**2
            
            if ab_len_sq == 0:
                return a
                
            dy_ap = p[0] - a[0]
            dx_ap = (p[1] - a[1]) * lon_scale
            
            t = (dy_ap * dy_ab + dx_ap * dx_ab) / ab_len_sq
            t = max(0, min(1, t)) # Clamp to segment limits
            
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

        for way in self.ways:
            for i in range(len(way) - 1):
                n1, n2 = way[i], way[i+1]
                if n1 in self.nodes and n2 in self.nodes:
                    p1 = self.nodes[n1]
                    p2 = self.nodes[n2]
                    
                    proj_p = closest_point_on_segment((lat, lon), p1, p2)
                    d_sq = dist_squared((lat, lon), proj_p)
                    
                    if d_sq < min_dist:
                        min_dist = d_sq
                        best_point = proj_p
                        
        return best_point

    def parse_osm_to_graph(self):
        """
        Converts the parsed OSM data into Node Features and Edge Index for GATv2.
        Returns:
            node_features: np.array of shape (num_nodes, feature_dim)
            edge_index: np.array of shape (2, num_edges)
        """
        if not self.road_nodes:
            return np.array([]), np.array([[], []])
            
        # Create a mapping from OSM node ID to a continuous integer index 0...N-1
        node_to_idx = {}
        idx = 0
        for node_id in self.road_nodes:
            if node_id in self.nodes:
                node_to_idx[node_id] = idx
                idx += 1
                
        num_nodes = idx
        node_features = np.zeros((num_nodes, 2), dtype=np.float32) # Features: [lat, lon]
        
        for node_id, node_idx in node_to_idx.items():
            node_features[node_idx] = self.nodes[node_id]
            
        # Normalize features (Optional, but good for GNNs)
        # We can center the coordinates around the mean
        if num_nodes > 0:
            mean_lat = np.mean(node_features[:, 0])
            mean_lon = np.mean(node_features[:, 1])
            node_features[:, 0] -= mean_lat
            node_features[:, 1] -= mean_lon
            # Scale
            max_val = np.max(np.abs(node_features))
            if max_val > 0:
                node_features /= max_val
            
        # Build Edge Index
        src = []
        dst = []
        
        for way in self.ways:
            # Connect sequential nodes in the way
            for i in range(len(way) - 1):
                u = way[i]
                v = way[i+1]
                
                if u in node_to_idx and v in node_to_idx:
                    u_idx = node_to_idx[u]
                    v_idx = node_to_idx[v]
                    
                    # Assume undirected graph for road topology simplicity 
                    # (we can parse one-way tags later if needed)
                    src.append(u_idx)
                    dst.append(v_idx)
                    src.append(v_idx)
                    dst.append(u_idx)
                    
        edge_index = np.array([src, dst], dtype=np.int64)
        return node_features, edge_index
