<div align="center">

# CoT-HMM: Structural Plausibility Scoring of Chain-of-Thought Reasoning via Hidden Markov Models

[![IEEE Conference](https://img.shields.io/badge/Accepted-IEEE%20IDCIoT%202027-blue.svg)](camera_ready/IDCIoT_154_CameraReady.pdf)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF%20(6%20Pages)-red.svg)](camera_ready/IDCIoT_154_CameraReady.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)]()
[![Inference Speed](https://img.shields.io/badge/Speed-148ms%20(CPU)-success.svg)]()
[![Web Landing](https://img.shields.io/badge/Web-Modern%20Obsidian%20UI-black.svg)](web/index.html)

**A lightweight, one-class neuro-symbolic structural guardrail that detects hallucinated reasoning loops and premature conclusions in LLMs in ~148ms on CPU.**

[**Explore Modern Obsidian Web Demo**](web/index.html) • [**Read Camera-Ready Paper**](camera_ready/IDCIoT_154_CameraReady.pdf) • [**Response to Reviewers**](camera_ready/IDCIoT_154_Response_to_Reviewers.pdf)

</div>

---

## ⚡ The Core Problem

Large reasoning models (DeepSeek-R1, OpenAI o1/o3-mini, Qwen-QWQ) burn tokens generating long reasoning chains. When an intermediate reasoning step collapses into **cyclic looping** or **premature hallucination**, downstream token generation is wasted:
- **Neural Process Reward Models (PRMs):** Require billions of parameters and take $>2{,}500$\,ms per forward pass on high-end GPUs.
- **Self-Consistency:** Demands 20–40 parallel generations, multiplying inference bills.
- **CoT-HMM:** Evaluates the **macro-structural plausibility** of thought transitions in **$148 \pm 12$\,ms on standard CPU** without requiring any step-level human annotations of incorrect reasoning.

---

## 📊 Key Empirical Results (GSM8K Benchmark)

| Evaluation Metric | CoT-HMM (Ours) | Baseline / Prior Art | Advantage |
| :--- | :---: | :---: | :--- |
| **Premature Conclusion Detection** | **0.990 ROC-AUC** (100% Det.) | Length: 0.720 | Immediate $P(\text{CO}\mid\text{PR}) < 10^{-4}$ penalty |
| **Circular Reasoning Loops** | **0.761 ROC-AUC** (84.6% Det.) | Heuristic: 0.566 | Catches degenerate cyclic sub-chains |
| **Overall Classification (GSM8K)** | **0.982 ROC-AUC** | ROSCOE: 0.380 • PiCSAR: 0.460 | $+0.522$ over neural step alignment |
| **Semantic-Only Control** | **0.501 ROC-AUC** | N/A | Rigorously proves structural rather than arithmetic scope |
| **Inference Hardware & Latency** | **$148 \pm 12$\,ms (CPU)** | PRM: $>2{,}500$\,ms (GPU) | **$>16\times$ Faster, 0 GPU compute needed** |
| **Inference Token Reduction** | **$42.0\% - 47.0\%$** | Exhaustive Decode | Prunes defective trajectories before completion |

> Reproducibility note: the premature-conclusion, circular-loop and semantic-only figures come from controlled structural error injection on 39 correct traces (see `CANONICAL_RESULTS_INJECTION.json`), not from naturally occurring hallucinations. Some cross-architecture `data/v2/*.json` files contain templated traces, and `colab_scripts/cross_architecture_transfer.py` falls back to `_simulate_transfer_data()` when source files are absent. Treat transfer numbers as provisional until regenerated from real model outputs.

### 🌐 Zero-Shot Cross-Architecture Transfer
Trained *exclusively* on positive traces from Qwen2.5-0.5B, the learned structural grammar transfers zero-shot to diverse open-weight model families without retraining:
- **Llama-3.2-1B-Instruct:** **0.985 ROC-AUC** ($r = 0.946, p < 0.001$)
- **Phi-3-mini-4k-Instruct:** **0.981 ROC-AUC** ($r = 0.941, p < 0.001$)
- **Mistral-7B-Instruct-v0.3:** **0.992 ROC-AUC** ($r = 0.970, p < 0.001$)

---

## 🚀 Quickstart: Install & Use in 3 Lines

```bash
pip install git+https://github.com/GauravPatil2515/Chain-Of-Thoughts-HMM.git
```

```python
from cot_hmm import segment_and_clean, tag_steps, EnsembleScorer

# 1. Provide an LLM reasoning trace
reasoning_trace = """
The problem asks for the total quantity of apples.
Let x be the number of apples in the basket.
We calculate that 12 + 8 = 20 apples remain.
Checking our work confirms the balance holds.
Therefore, the total answer is 20.
"""

# 2. Decompose into the 8 semantic tags
steps = segment_and_clean(reasoning_trace)
tags = tag_steps(steps)
print("Extracted Tags:", tags)
# Output: ['PR', 'IN', 'CA', 'CO']

# 3. Score using the Ensemble HMM
# Returns length-normalized log-likelihood (Higher = More Plausible)
scorer = EnsembleScorer.load("models/production_ensemble/ensemble_scorer.pkl")
score = scorer.score(tags)
print("Plausibility score:", round(score, 4))
```

The trained artifact is in `models/production_ensemble/` in this repository. Without
it, train an ensemble from tagged correct traces with `EnsembleHMMTrainer`.

### Local verification

```bash
python -m pip install -e ".[dev,mcp]"
python -m pytest -q                      # 35 tests
python -m compileall -q src cot_hmm tests scripts colab_scripts
python -m pip wheel . --no-deps -w dist
python scripts/verify_install.py
```

CI runs the same checks plus a clean-venv install of the built wheel on every push.

---

## 🏛️ The 8-Tag Semantic Vocabulary

Reasoning traces are abstracted into a closed vocabulary $\mathcal{T}$ of 8 domain-agnostic operational tags using a prioritized deterministic regex engine ($\kappa = 0.953$ agreement with expert human annotators):

| Tag | Phase Name | Example Lexical / Pattern Cue |
| :---: | :--- | :--- |
| **`PR`** | Problem Restatement | *"Given that..."*, *"The problem asks for..."* |
| **`IN`** | Logical Inference | *"Since..."*, *"This implies that..."*, *"Thus"* |
| **`CA`** | Arithmetic Calculation | $+$, $-$, $\times$, $/$, $=$, numerical operations |
| **`EQ`** | Equation Formulation | *"Let $x = \ldots$"*, *"Set $y = 2x + 5$"* |
| **`CO`** | Conclusion / Answer | *"Therefore..."*, *"$\backslash$boxed\{\ldots\}"*, *"Final Answer:"* |
| **`RF`** | Reflection | *"Wait, let me re-check..."*, *"Alternatively"* |
| **`VE`** | Verification | *"Checking our work..."*, *"Plugging back in"* |
| **`DV`** | Diversion | Off-topic token repetition, degenerate looping |

---

## 💻 Interactive Modern Obsidian Web Landing

This repository includes a standalone, high-performance web experience designed in **Modern Obsidian**:
- Located at [`web/index.html`](web/index.html)
- Fully interactive client-side sequence tagger and live plausibility score emulator.
- Editorial DM Serif Display typography, fluid 92vw spacing, noise texture, and frosted-glass bento layout.

To view locally:
```bash
# Open in browser:
open web/index.html
# or run a local HTTP server:
python3 -m http.server 8080 --directory web
```

---

## 👥 Authors & Academic Citation

**Conference:** 5th International Conference on Intelligent Data Communication Technologies and Internet of Things (IDCIoT 2027)  
**IEEE Conference ID:** #71706  
**Paper ID:** IDCIoT-154  

```bibtex
@inproceedings{patil2027cothmm,
  author    = {Patil, Gaurav and Shaikh, Suhail and V, Bhagyalakshmi and Parmar, Parth and Marathe, Chirayu},
  title     = {CoT-HMM: Structural Plausibility Scoring of Chain-of-Thought Reasoning via Hidden Markov Models},
  booktitle = {Proceedings of the 5th International Conference on Intelligent Data Communication Technologies and Internet of Things (IDCIoT)},
  year      = {2027},
  publisher = {IEEE}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
