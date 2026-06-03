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
# File: ui/map_selector.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import tkinter as tk
from tkinter import ttk, messagebox
from ui.translator import translator

try:
    import tkintermapview
except ImportError:
    tkintermapview = None

class MapSelectorWindow(tk.Toplevel):
    def __init__(self, parent, map_provider, num_cameras, num_loops, on_complete_callback):
        super().__init__(parent)
        
        self.title(translator.t("map_title"))
        self.geometry("800x600")
        
        if tkintermapview is None:
            messagebox.showerror(translator.t("err_title"), translator.t("map_err_tkmap"))
            self.destroy()
            return
            
        self.map_provider = map_provider
        self.num_cameras = num_cameras
        self.num_loops = num_loops
        self.on_complete_callback = on_complete_callback
        
        self.selected_cameras = []
        self.selected_loops = []
        
        # State: 0 = placing cameras, 1 = placing loops, 2 = done
        self.state = 0 if self.num_cameras > 0 else (1 if self.num_loops > 0 else 2)
        
        self.setup_ui()
        self.center_map()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        if not getattr(self, 'confirmed', False):
            self.on_complete_callback(None, None)
        self.destroy()
        
    def setup_ui(self):
        # Top panel for instructions
        self.top_frame = ttk.Frame(self, padding="10")
        self.top_frame.pack(fill="x")
        
        self.lbl_instructions = ttk.Label(self.top_frame, text="", font=("Helvetica", 14, "bold"))
        self.lbl_instructions.pack(side="left")
        
        self.btn_confirm = ttk.Button(self.top_frame, text=translator.t("map_btn_confirm"), command=self.confirm_and_close, state="disabled")
        self.btn_confirm.pack(side="right")
        
        self.update_instructions()
        
        # Map widget
        self.map_widget = tkintermapview.TkinterMapView(self, corner_radius=0)
        self.map_widget.pack(fill="both", expand=True)
        
        # Right click to add marker
        self.map_widget.add_right_click_menu_command(label="Place Sensor", command=self.add_marker, pass_coords=True)

    def center_map(self):
        bounds = self.map_provider.get_bounds()
        if bounds:
            min_lat, min_lon, max_lat, max_lon = bounds
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            
            # Roughly adjust zoom based on bounds
            # This is a basic heuristic.
            self.map_widget.set_position(center_lat, center_lon)
            self.map_widget.set_zoom(13)
        else:
            self.map_widget.set_position(-23.3, -51.1) # Default Londrina
            self.map_widget.set_zoom(10)

    def update_instructions(self):
        if self.state == 0:
            remaining = self.num_cameras - len(self.selected_cameras)
            self.lbl_instructions.config(text=translator.t("map_inst_camera", remaining))
            if remaining == 0:
                self.state = 1 if self.num_loops > 0 else 2
                self.update_instructions()
        elif self.state == 1:
            remaining = self.num_loops - len(self.selected_loops)
            self.lbl_instructions.config(text=translator.t("map_inst_loop", remaining))
            if remaining == 0:
                self.state = 2
                self.update_instructions()
        elif self.state == 2:
            self.lbl_instructions.config(text=translator.t("map_inst_done"))
            self.btn_confirm.config(state="normal")

    def add_marker(self, coords):
        if self.state == 2:
            return
            
        raw_lat, raw_lon = coords
        # Snap to the closest road from the OSM topology
        snapped_lat, snapped_lon = self.map_provider.snap_to_road(raw_lat, raw_lon)
        
        if self.state == 0:
            self.selected_cameras.append({"lat": snapped_lat, "lon": snapped_lon})
            self.map_widget.set_marker(snapped_lat, snapped_lon, text=f"Camera {len(self.selected_cameras)}", marker_color_outside="red", marker_color_circle="darkred")
        elif self.state == 1:
            self.selected_loops.append({"lat": snapped_lat, "lon": snapped_lon})
            self.map_widget.set_marker(snapped_lat, snapped_lon, text=f"Loop {len(self.selected_loops)}", marker_color_outside="blue", marker_color_circle="darkblue")
            
        self.update_instructions()

    def confirm_and_close(self):
        self.confirmed = True
        self.on_complete_callback(self.selected_cameras, self.selected_loops)
        self.destroy()
