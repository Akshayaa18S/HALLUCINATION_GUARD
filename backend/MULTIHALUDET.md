# MultiHaluDet branch

This backend now has **two independent hallucination-detection signals**,
fused into the final decision:

```
Signal 1 - INTERNAL (multihaludet/)
  "What does the LLM's own hidden-state trajectory suggest?"
  Frozen LLM -> hidden-state/logit extraction -> dynamic multi-depth
  layer sampling -> multi-scale attention -> layer-weighted Transformer
  encoder -> self-attention pooling -> global branch -> gated fusion ->
  out-of-fold ensemble meta-learner -> internal_hallucination_probability

Signal 2 - EXTERNAL (hallucination/, retrieval/, services/verification_service.py)
  "Does external evidence support the generated claims?"
  Claim extraction -> evidence retrieval (Wikipedia/FEVER) -> claim
  verification -> supported/contradicted/insufficient

                    FUSION (services/pipeline_service.py::_fuse_internal_and_external)
                    weighted average, weight = settings.fusion_internal_weight
                                 |
                                 v
                    final hallucination_result
```

## What changed and why

The base paper's method (multi-layer hidden-state trajectory probing)
needs access to a model's internal activations at multiple depths during
generation. The backend previously called an LLM only through **Ollama's
HTTP API** (`services/ollama_service.py` / `services/llm_service.py`),
which returns final text only - there is no way to get per-layer hidden
states out of Ollama. Stage 2 (`_stage_2`) of the live pipeline
(`services/pipeline_service.py`) now generates via a **local, in-process
HuggingFace model** (`multihaludet/generation_backend.py`) instead, so
the same forward pass that produces the answer also yields the hidden
states stages 3-4 need. Ollama is still used by
`services/verification_service.py` for the (much cheaper, non-hidden-state)
claim-verification JSON prompts in the external branch - that call didn't
need to move.

Stages 3 ("Hidden State Extraction") and 4 ("Feature Extraction") in
`services/pipeline_service.py` were previously **placeholder arithmetic**
on token counts (e.g. `token_count / 10`) - not real hidden states at
all. They now call the real `multihaludet/` pipeline.

## Package layout

```
multihaludet/
  generation_backend.py   Local HF model: generate() + capture per-layer
                           hidden states and per-step logits in one pass.
  layer_sampling.py        Dynamic multi-depth layer sampling; builds the
                            sequential representation (per-layer step
                            trajectories) and the global feature vector
                            (top-k probs, margins, entropy, logit stats,
                            layer-norm trajectory, anchor descriptors,
                            interaction features). Pure numpy.
  attention.py              MultiScaleAttention, LayerWeightedTransformer-
                             Encoder, SelfAttentionPooling. torch nn.Modules.
  fusion.py                 GlobalBranch (MLP) + GatedFusion.
  ensemble.py                Out-of-fold ensemble meta-learner (5 base-
                              learner heads + a logistic-regression-style
                              meta layer). See its module docstring for an
                              important caveat about what "trained" means
                              here.
  pipeline.py                MultiHaluDetModel: wires all of the above
                              into one forward(bundle) -> dict.
  service.py                  multihaludet_service singleton: the only
                               thing the rest of the backend imports.
  training/
    datasets.py               HaluEval / TriviaQA / multilingual
                               (French, Bangla, Amharic) loader interfaces.
    train.py, evaluate.py     OOF training / evaluation scripts.
```

## Setup

```bash
pip install -r requirements.txt   # now includes torch, transformers, accelerate, scikit-learn
```

Configure in `.env` (all optional, defaults shown in
`config/settings.py`):

```
MULTIHALUDET_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct   # any HF causal LM
MULTIHALUDET_DEVICE=cpu                              # cpu | cuda | mps
MULTIHALUDET_CHECKPOINT_PATH=./multihaludet/checkpoints/multihaludet.pt
FUSION_INTERNAL_WEIGHT=0.5
```

The first request will download the configured model from Hugging Face
and load it into memory - this needs network access and enough RAM/VRAM
for the model you pick; a 0.5B-1.5B instruct model is a reasonable
default for CPU.

## Important limitation: the ensemble is currently untrained

`multihaludet/pipeline.py`'s `MultiHaluDetModel` is architecturally
complete and runs end-to-end, but **its weights are randomly
initialized** until you train it. Every internal score it returns is
tagged `is_trained: false` (surfaced in the API's stage 4 metadata and
stage 7 explainability output) so this is never silently hidden.

To get a real detector:

1. Get labeled data: HaluEval (qa/dialogue/summarization) and/or
   labeled TriviaQA responses - see `multihaludet/training/datasets.py`
   for expected file formats and download pointers. Add French/Bangla/
   Amharic splits for the paper's cross-lingual generalization claim.
2. `HFGenerationBackend.score_existing_response(query, response)` is now
   implemented - a **teacher-forced** single forward pass
   (`output_hidden_states=True` over the concatenated prompt+response
   sequence, no `model.generate()` call), so `train.py`/`evaluate.py`
   score the dataset's actual labeled response rather than a freshly
   generated one. `pipeline.py::MultiHaluDetModel` exposes a matching
   grad-enabled path (`compute_deep_features` / `predict_from_features`)
   that both training scripts use; `forward()` itself stays no_grad for
   inference. See `multihaludet/tests/` for unit coverage of both.
3. Run `python -m multihaludet.training.train --halueval-qa ... --triviaqa ...`
4. Point `multihaludet_checkpoint_path` at the resulting checkpoint.

None of this was run end-to-end in this environment - no network access
to download the base model or datasets, no GPU, and no labeled data on
disk. What *is* verified here: `multihaludet/layer_sampling.py`'s numpy
math (no torch dependency), and - via `multihaludet/tests/` with a tiny
mocked HF model - that `score_existing_response` preserves the fixed
response, returns correctly-shaped hidden states, that gradients reach
every trainable MultiHaluDet component through `compute_deep_features`,
and that checkpoint save/load and `is_trained` behave correctly. Running
the full loop against real HaluEval/TriviaQA data with a real downloaded
model still needs to happen in an environment with network + GPU.

## Fusion

`_fuse_internal_and_external` in `services/pipeline_service.py` combines
the two signals with a single transparent weight
(`fusion_internal_weight`, default 0.5) rather than a second opaque
model, and discounts confidence when the two signals disagree. Until the
internal branch is trained, you likely want to lower
`fusion_internal_weight` (or set it to 0) so the fused decision leans on
the validated external/RAG signal rather than the untrained one.
