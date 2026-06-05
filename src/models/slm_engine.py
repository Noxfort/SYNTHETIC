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
# File: models/slm_engine.py
# Author: Gabriel Moraes
# Date: 2026-02-26

import gc
import os
import re
import sys
from typing import Optional
import numpy as np

from src.core.logger import logger

# Attempt to import the inference library
try:
    from llama_cpp import Llama
except ImportError:
    logger.critical("llama-cpp-python not installed. Please run: pip install llama-cpp-python")
    sys.exit(1)

class SLMEngine:
    """
    Wrapper class for the Phi-4-mini GGUF Model optimized for Vector output.
    It focuses on extracting semantic embeddings (Latent Vectors) rather than just text.
    """

    # Context window set to 8192 to allow deep cognitive reasoning (Thinking Mode)
    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 8192, n_gpu_layers: int = 0, temperature: float = 0.85) -> None:
        """
        Initializes the inference engine with Embedding support enabled.
        """
        self.temperature: float = temperature
        if model_path is None:
            # Automatic path resolution (lowercase 'vault')
            base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path: str = os.path.join(base_dir, "models", "vault", "Phi-4-mini-reasoning-UD-Q6_K_XL.gguf")
        else:
            self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[SLMEngine] Model file not found at: {self.model_path}\n"
                "Please run 'download_qwen.py' first or move the model to the correct directory."
            )

        logger.info(f"[SLMEngine] Loading model for Vector Synthesis from: {self.model_path}...")
        logger.info(f"[SLMEngine] Context Window Capacity: {n_ctx} tokens.")

        try:
            self.llm: Optional[Llama] = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                embedding=True  # CRITICAL: Enables vector extraction
            )
            logger.info("[SLMEngine] Vector Engine loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"[SLMEngine] Failed to initialize Llama context: {e}") from e

    def dream_scenario_vector(self, system_instruction: str, user_prompt: str) -> list:
        """
        Generates a creative scenario internally (Transient Memory) and returns ONLY its 
        numerical vector representation (Embedding).
        
        The text is generated, encoded, and immediately discarded after being printed.
        
        Args:
            system_instruction (str): The persona/rules.
            user_prompt (str): The specific day context.

        Returns:
            list[float]: A high-dimensional vector representing the 'feeling' of the scenario.
        """
        
        # Injecting the Cognitive Directive to force Chain-of-Thought (Thinking Mode)
        cognitive_instruction: str = (
            system_instruction + 
            "\n\n[CRITICAL DIRECTIVE]: You are in THINKING MODE. "
            "Before providing the final scenario, you MUST open a <think> tag and write out your step-by-step "
            "logical reasoning about the weather, traffic density, and physical constraints of the requested situation. "
            "Close with </think> and then write the final scenario description."
        )

        # 1. Generate the creative text (The "Dream") in memory
        messages: list = [
            {"role": "system", "content": cognitive_instruction},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Temperature defined by UI mode, expanded max_tokens for deep thoughts
            assert self.llm is not None
            completion = self.llm.create_chat_completion(
                messages=messages,
                temperature=self.temperature, 
                max_tokens=1024
            )
            dream_text: str = completion['choices'][0]['message']['content']
            
            # Strip <think>...</think> for display — thinking is internal only
            # The full text (with thinking) is still used for embedding generation
            visible_text: str = re.sub(r'<think>.*?</think>', '', dream_text, flags=re.DOTALL).strip()
            
            logger.info("\n" + "="*50)
            logger.info("Scenario Description:")
            logger.info("-" * 50)
            logger.info(visible_text if visible_text else "[Thinking mode active — scenario encoded directly]")
            logger.info("="*50 + "\n")
            
            # 2. Convert the Dream (Thoughts + Scenario) to a Vector (Embedding)
            logger.info("[SLMEngine] Converting thoughts into Latent Vector...")
            embedding_response = self.llm.create_embedding(dream_text)
            
            # Extract the raw vector list
            raw_vector = embedding_response['data'][0]['embedding']
            
            # Qwen3 4B latent dimension target (keeping 2048 for CSDI compatibility)
            latent_dim: int = 2048
            vector: list
            
            # Robust Mean Pooling to handle variable llama.cpp output structures
            if isinstance(raw_vector[0], list):
                # Output is a list of token vectors: [[...], [...], ...]
                logger.info(f"[SLMEngine] Nested token embeddings detected. Pooling {len(raw_vector)} tokens...")
                vector_matrix = np.array(raw_vector)
                vector = vector_matrix.mean(axis=0).tolist()
            else:
                # Output is a flat list
                if len(raw_vector) > latent_dim and len(raw_vector) % latent_dim == 0:
                    # Flattened token array
                    logger.info(f"[SLMEngine] Flat token embeddings detected. Pooling...")
                    num_tokens = len(raw_vector) // latent_dim
                    vector_matrix = np.array(raw_vector).reshape(num_tokens, latent_dim)
                    vector = vector_matrix.mean(axis=0).tolist()
                else:
                    # Already a single vector (or needs truncation)
                    vector = raw_vector[:latent_dim]
            
            # Failsafe padding or truncation just to guarantee the exact shape for VAE-TCN
            if len(vector) < latent_dim:
                vector = vector + [0.0] * (latent_dim - len(vector))
            elif len(vector) > latent_dim:
                vector = vector[:latent_dim]
            
            return vector
            
        except Exception as e:
            logger.error(f"[SLMEngine] Vector Synthesis Error: {e}", exc_info=True)
            # Return a zero-vector fallback for 4B models
            return [0.0] * 2048

    def release(self) -> None:
        """
        Fully releases the LLM from memory to free resources for other models.
        After calling this, the engine cannot generate new vectors until reloaded.
        This is a heavier model (~4GB) — releasing it before CSDI/VAE-TCN
        is critical for machines with limited memory.
        """
        if hasattr(self, 'llm') and self.llm is not None:
            logger.info("[SLMEngine] Releasing LLM from memory...")
            del self.llm
            self.llm = None
            gc.collect()
            logger.info("[SLMEngine] LLM released successfully. ~4GB freed.")
        else:
            logger.info("[SLMEngine] LLM already released, nothing to free.")

# Self-test block
if __name__ == "__main__":
    try:
        print("Testing Vector Generation with Enhanced Cognitive Context...")
        engine = SLMEngine()
        vector = engine.dream_scenario_vector(
            "You are an expert traffic scenario simulator.", 
            "Imagine a chaotic stormy Monday morning."
        )
        print(f"\nGenerated Vector Dimension: {len(vector)}")
        print(f"First 5 dimensions: {vector[:5]}...")
    except Exception as err:
        print(f"\nTest Failed: {err}")