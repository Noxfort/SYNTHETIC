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
# File: ui/gui.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from src.core.map_provider import OSMMapProvider
from ui.map_selector import MapSelectorWindow
from ui.translator import translator

class DataGeneratorApp(tk.Tk):
    """
    The main Graphical User Interface (GUI) for the Data Generator.
    This module handles only the visual components and user inputs.
    It passes the configuration to the orchestrator via a callback.
    """
    def __init__(self, on_generate_callback=None):
        super().__init__()

        self.on_generate_callback = on_generate_callback
        
        # --- State Variables ---
        self.var_waze = tk.BooleanVar(value=True)
        self.var_tomtom = tk.BooleanVar(value=True)
        self.var_loop = tk.BooleanVar(value=True)
        self.var_camera = tk.BooleanVar(value=True)

        self.var_gaps = tk.BooleanVar(value=True)
        self.var_anomalies = tk.BooleanVar(value=True)

        self.var_duration = tk.IntVar(value=1) 
        self.var_interval = tk.IntVar(value=10)
        self.var_flow_level = tk.StringVar(value="Médio") 
        self.var_slm_mode = tk.StringVar(value="Realista")

        self.var_num_cameras = tk.IntVar(value=5)
        self.var_num_loops = tk.IntVar(value=5)
        
        # Set default output directory
        default_output = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.var_output_dir = tk.StringVar(value=default_output)
        
        self.var_osm_path = tk.StringVar(value="")

        # Language mapping for the combobox
        self.lang_map = {
            "English": "en",
            "Português (Brasil)": "pt-br",
            "Français": "fr",
            "中文 (简体)": "zh-cn",
            "Русский": "ru",
            "Español": "es"
        }
        self.reverse_lang_map = {v: k for k, v in self.lang_map.items()}

        # --- GUI Layout ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        # --- Language Selection ---
        self.lang_frame = ttk.Frame(main_frame)
        self.lang_frame.pack(fill="x", pady=(0, 10))
        self.lbl_lang = ttk.Label(self.lang_frame, text="")
        self.lbl_lang.pack(side="left", padx=5)
        
        self.combo_lang = ttk.Combobox(self.lang_frame, values=list(self.lang_map.keys()), state="readonly")
        self.combo_lang.pack(side="left")
        self.combo_lang.bind("<<ComboboxSelected>>", self.on_language_change)
        
        # Set default combobox value based on current translator locale
        self.combo_lang.set(self.reverse_lang_map.get(translator.get_locale(), "English"))

        # --- Section 1: Sources ---
        self.sources_frame = ttk.LabelFrame(main_frame, padding="10")
        self.sources_frame.pack(fill="x", expand=True)
        
        self.chk_waze = ttk.Checkbutton(self.sources_frame, variable=self.var_waze)
        self.chk_waze.pack(anchor="w")
        self.chk_tomtom = ttk.Checkbutton(self.sources_frame, variable=self.var_tomtom)
        self.chk_tomtom.pack(anchor="w")
        self.chk_loop = ttk.Checkbutton(self.sources_frame, variable=self.var_loop)
        self.chk_loop.pack(anchor="w")
        self.chk_camera = ttk.Checkbutton(self.sources_frame, variable=self.var_camera)
        self.chk_camera.pack(anchor="w")

        # --- Section 2: Problems ---
        self.problems_frame = ttk.LabelFrame(main_frame, padding="10")
        self.problems_frame.pack(fill="x", expand=True, pady=5)
        
        self.chk_gaps = ttk.Checkbutton(self.problems_frame, variable=self.var_gaps)
        self.chk_gaps.pack(anchor="w")
        self.chk_anomalies = ttk.Checkbutton(self.problems_frame, variable=self.var_anomalies)
        self.chk_anomalies.pack(anchor="w")

        # --- Section 3: Settings ---
        self.config_frame = ttk.LabelFrame(main_frame, padding="10")
        self.config_frame.pack(fill="x", expand=True, pady=5)
        
        # Duration
        dur_frame = ttk.Frame(self.config_frame)
        self.lbl_duration = ttk.Label(dur_frame)
        self.lbl_duration.pack(side="left", padx=5)
        ttk.Entry(dur_frame, textvariable=self.var_duration, width=10).pack(side="left")
        dur_frame.pack(anchor="w")
        
        # Interval
        int_frame = ttk.Frame(self.config_frame)
        self.lbl_interval = ttk.Label(int_frame)
        self.lbl_interval.pack(side="left", padx=5)
        ttk.Entry(int_frame, textvariable=self.var_interval, width=10).pack(side="left")
        int_frame.pack(anchor="w", pady=5)

        # Flow Level
        flow_frame = ttk.Frame(self.config_frame)
        self.lbl_flow_level = ttk.Label(flow_frame)
        self.lbl_flow_level.pack(side="left", padx=5)
        self.rad_flow_small = ttk.Radiobutton(flow_frame, variable=self.var_flow_level, value="Pequeno")
        self.rad_flow_small.pack(side="left")
        self.rad_flow_med = ttk.Radiobutton(flow_frame, variable=self.var_flow_level, value="Médio")
        self.rad_flow_med.pack(side="left")
        self.rad_flow_large = ttk.Radiobutton(flow_frame, variable=self.var_flow_level, value="Grande")
        self.rad_flow_large.pack(side="left")
        self.rad_flow_chaotic = ttk.Radiobutton(flow_frame, variable=self.var_flow_level, value="Caótico")
        self.rad_flow_chaotic.pack(side="left")
        flow_frame.pack(anchor="w", pady=5)
        
        # SLM Mode
        slm_frame = ttk.Frame(self.config_frame)
        self.lbl_slm_mode = ttk.Label(slm_frame)
        self.lbl_slm_mode.pack(side="left", padx=5)
        self.rad_slm_ultra = ttk.Radiobutton(slm_frame, variable=self.var_slm_mode, value="Ultrarealista")
        self.rad_slm_ultra.pack(side="left")
        self.rad_slm_real = ttk.Radiobutton(slm_frame, variable=self.var_slm_mode, value="Realista")
        self.rad_slm_real.pack(side="left")
        self.rad_slm_creative = ttk.Radiobutton(slm_frame, variable=self.var_slm_mode, value="Criativo")
        self.rad_slm_creative.pack(side="left")
        slm_frame.pack(anchor="w", pady=5)
        
        # Local Sensor Configuration
        local_sensors_frame = ttk.Frame(self.config_frame)
        self.lbl_num_cameras = ttk.Label(local_sensors_frame)
        self.lbl_num_cameras.pack(side="left", padx=5)
        ttk.Entry(local_sensors_frame, textvariable=self.var_num_cameras, width=5).pack(side="left", padx=5)
        
        self.lbl_num_loops = ttk.Label(local_sensors_frame)
        self.lbl_num_loops.pack(side="left", padx=5)
        ttk.Entry(local_sensors_frame, textvariable=self.var_num_loops, width=5).pack(side="left", padx=5)
        local_sensors_frame.pack(anchor="w", pady=5)

        # --- Section 4: Output Folder ---
        self.output_frame = ttk.LabelFrame(main_frame, padding="10")
        self.output_frame.pack(fill="x", expand=True, pady=5)
        
        self.output_label = ttk.Label(self.output_frame, textvariable=self.var_output_dir, relief="sunken")
        self.output_label.pack(fill="x", side="left", expand=True, padx=5)
        self.btn_select_output = ttk.Button(self.output_frame, command=self.select_output_dir)
        self.btn_select_output.pack(side="left")

        # --- Section 4.5: Map File ---
        self.map_frame = ttk.LabelFrame(main_frame, padding="10")
        self.map_frame.pack(fill="x", expand=True, pady=5)
        
        self.osm_label = ttk.Label(self.map_frame, textvariable=self.var_osm_path, relief="sunken")
        self.osm_label.pack(fill="x", side="left", expand=True, padx=5)
        self.btn_browse_osm = ttk.Button(self.map_frame, command=self.select_osm_file)
        self.btn_browse_osm.pack(side="left")

        # --- Section 5: Action ---
        self.action_button = ttk.Button(main_frame, command=self.start_generation)
        self.action_button.pack(fill="x", expand=True, ipady=10, pady=10)

        # Apply translations
        self.update_ui_texts()

    def on_language_change(self, event=None):
        selected = self.combo_lang.get()
        new_locale = self.lang_map.get(selected)
        if new_locale:
            translator.set_locale(new_locale)
            self.update_ui_texts()

    def update_ui_texts(self):
        """Updates all text in the UI dynamically via the Translator."""
        t = translator.t
        self.title(t("app_title"))
        self.lbl_lang.config(text=t("lang_label"))
        
        self.sources_frame.config(text=t("section_sources"))
        self.chk_waze.config(text=t("waze"))
        self.chk_tomtom.config(text=t("tomtom"))
        self.chk_loop.config(text=t("loop"))
        self.chk_camera.config(text=t("camera"))

        self.problems_frame.config(text=t("section_problems"))
        self.chk_gaps.config(text=t("gaps"))
        self.chk_anomalies.config(text=t("anomalies"))

        self.config_frame.config(text=t("section_settings"))
        self.lbl_duration.config(text=t("duration"))
        self.lbl_interval.config(text=t("interval"))
        self.lbl_flow_level.config(text=t("flow_level"))
        
        self.rad_flow_small.config(text=t("flow_small"))
        self.rad_flow_med.config(text=t("flow_medium"))
        self.rad_flow_large.config(text=t("flow_large"))
        self.rad_flow_chaotic.config(text=t("flow_chaotic"))

        self.lbl_slm_mode.config(text=t("slm_mode"))
        self.rad_slm_ultra.config(text=t("slm_ultra"))
        self.rad_slm_real.config(text=t("slm_real"))
        self.rad_slm_creative.config(text=t("slm_creative"))

        self.lbl_num_cameras.config(text=t("num_cameras"))
        self.lbl_num_loops.config(text=t("num_loops"))

        self.output_frame.config(text=t("section_output"))
        self.btn_select_output.config(text=t("btn_select"))

        self.map_frame.config(text=t("section_map"))
        self.btn_browse_osm.config(text=t("btn_browse_osm"))

        if str(self.action_button.cget("state")) != "disabled":
            self.action_button.config(text=t("btn_generate"))

    def select_output_dir(self):
        directory = filedialog.askdirectory(initialdir=self.var_output_dir.get())
        if directory:
            self.var_output_dir.set(directory)
            
    def select_osm_file(self):
        filepath = filedialog.askopenfilename(
            title=translator.t("section_map"),
            filetypes=[("OpenStreetMap", "*.osm"), ("All files", "*.*")]
        )
        if filepath:
            self.var_osm_path.set(filepath)

    def start_generation(self):
        osm_path = self.var_osm_path.get()
        if not osm_path or not os.path.exists(osm_path):
            messagebox.showerror(translator.t("err_title"), translator.t("err_osm_missing"))
            return

        self.action_button.config(text=translator.t("btn_loading_map"), state="disabled")
        
        try:
            map_provider = OSMMapProvider()
            map_provider.parse_osm_file(osm_path)
        except Exception as e:
            messagebox.showerror(translator.t("err_title"), translator.t("err_osm_failed", str(e)))
            self.action_button.config(text=translator.t("btn_generate"), state="normal")
            return
            
        MapSelectorWindow(
            self, 
            map_provider, 
            self.var_num_cameras.get(), 
            self.var_num_loops.get(),
            lambda cams, loops: self._continue_generation_after_map(cams, loops, map_provider)
        )

    def _continue_generation_after_map(self, cameras, loops, map_provider):
        if cameras is None or loops is None:
            self.action_button.config(text=translator.t("btn_generate"), state="normal")
            return
            
        self.action_button.config(text=translator.t("btn_generating"), state="disabled")
        
        base_output_dir = self.var_output_dir.get()
        final_output_dir = os.path.join(base_output_dir, "output")

        config = {
            "sources": {
                "waze": self.var_waze.get(),
                "tomtom": self.var_tomtom.get(),
                "loop": self.var_loop.get(),
                "camera": self.var_camera.get(),
            },
            "problems": {
                "gaps": self.var_gaps.get(),
                "anomalies": self.var_anomalies.get(),
            },
            "simulation": {
                "duration_days": self.var_duration.get(),
                "interval_seconds": self.var_interval.get(),
                "flow_level": self.var_flow_level.get(),
                "slm_mode": self.var_slm_mode.get(),
                "num_cameras": self.var_num_cameras.get(),
                "num_loops": self.var_num_loops.get()
            },
            "map": {
                "bounds": map_provider.get_bounds(),
                "provider": map_provider,
                "local_points": {
                    "cameras": cameras,
                    "loops": loops
                }
            },
            "output_directory": final_output_dir 
        }

        if self.on_generate_callback:
            self.on_generate_callback(config, self.handle_success, self.handle_error)
        else:
            self.handle_error(Exception("No generation callback provided!"))

    def handle_success(self, final_output_dir):
        self.after(0, self._on_generation_complete, final_output_dir)

    def handle_error(self, error):
        self.after(0, self._on_generation_error, error)

    def _on_generation_complete(self, final_output_dir):
        messagebox.showinfo(translator.t("success_title"), translator.t("success_msg", final_output_dir))
        self.action_button.config(text=translator.t("btn_generate"), state="normal")

    def _on_generation_error(self, error):
        print(f"Simulation error: {error}")
        messagebox.showerror(translator.t("err_title"), translator.t("err_generation", str(error)))
        self.action_button.config(text=translator.t("btn_generate"), state="normal")
