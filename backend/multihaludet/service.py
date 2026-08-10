"""
Public entry point for the MultiHaluDet branch. Everything else in the
backend should import `multihaludet_service` from here rather than
reaching into generation_backend.py / pipeline.py directly.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from multihaludet.generation_backend import GenerationBundle, HFGenerationBackend, hf_generation_backend

logger = logging.getLogger("hallucination_guard.multihaludet.service")


class MultiHaluDetService:
    def __init__(self, backend: HFGenerationBackend | None = None) -> None:
        self._backend = backend or hf_generation_backend
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from config import settings
            from multihaludet.pipeline import MultiHaluDetModel

            # Loading the model (to read hidden_size) happens lazily on
            # first use, same as generation itself.
            model = MultiHaluDetModel(
                hidden_size=self._backend.hidden_size,
                num_sampled_layers=settings.multihaludet_num_sampled_layers,
                attention_scales=list(settings.multihaludet_attention_scales),
                encoder_dim=settings.multihaludet_encoder_dim,
                encoder_heads=settings.multihaludet_encoder_heads,
                encoder_layers=settings.multihaludet_encoder_layers,
                global_top_k=settings.multihaludet_global_top_k,
                global_hidden_dim=settings.multihaludet_global_hidden_dim,
                ensemble_members=settings.multihaludet_ensemble_members,
            )
            model.eval()
            model.load_checkpoint(settings.multihaludet_checkpoint_path)
            self._model = model
            return self._model

    def is_available(self) -> bool:
        return self._backend.is_available()

    def generate(self, prompt: str, system: str | None = None) -> GenerationBundle:
        """Generation only (used by stage 2) - returns the bundle so
        stages 3/4 can extract features from it without regenerating."""
        return self._backend.generate_with_states(prompt, system=system)

    def score(self, bundle: GenerationBundle) -> dict[str, Any]:
        """Feature extraction + internal hallucination scoring (stages
        3-4) from an already-generated bundle."""
        model = self._ensure_model()
        return model(bundle)

    def generate_and_score(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """Convenience for callers that don't need the intermediate
        bundle (e.g. a training/eval script)."""
        bundle = self.generate(prompt, system=system)
        result = self.score(bundle)
        result["text"] = bundle.text
        return result


multihaludet_service = MultiHaluDetService()
