"""
MultiHaluDet base-paper branch: hidden-state trajectory probing for
internal hallucination detection.

Pipeline (mirrors the base paper's four stages):

    Frozen LLM + hidden-state/logit extraction   (generation_backend.py)
        -> Dynamic multi-depth layer sampling +
           sequential representation + global feature vector
                                                   (layer_sampling.py)
        -> Multi-scale attention                  (attention.py)
        -> Layer-weighted Transformer encoder      (attention.py)
        -> Self-attention pooling                  (attention.py)
        -> Global branch + gated fusion            (fusion.py)
        -> Out-of-fold deep features +
           ensemble meta-learner                   (ensemble.py)
        -> Internal hallucination probability

`pipeline.py` wires all of the above into a single `MultiHaluDetModel`.
`service.py` exposes the single object the rest of the backend imports
(`multihaludet_service`), combining the generation backend with the model
so a caller only ever needs `generate_and_score(prompt, system=None)`.

This branch is deliberately independent of the RAG/evidence-verification
branch (hallucination/, retrieval/, services/verification_service.py).
The two signals are combined afterwards - see
services/pipeline_service.py's `_fuse_internal_and_external` - not here.
"""

from multihaludet.service import multihaludet_service  # noqa: F401
