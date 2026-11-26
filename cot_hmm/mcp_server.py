#!/usr/bin/env python3
"""
CoT-HMM Model Context Protocol (MCP) Server.
Structural Plausibility Scoring & Hallucination Guardrail for Chain-of-Thought Reasoning.

Paper: "CoT-HMM: Structural Plausibility Scoring of Chain-of-Thought Reasoning via Hidden Markov Models"
Accepted at IEEE IDCIoT 2027 (Paper ID: IDCIoT-154)
Authors: Gaurav Patil et al.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from mcp.server.mcpserver import MCPServer
from cot_hmm import segment_and_clean, tag_steps, EnsembleScorer

# Initialize MCP Server
server = MCPServer("cot-hmm-guardrail")

# Constants & Semantic Tags
TAG_DEFINITIONS = {
    "PR": "Problem Restatement: Clarifying or rephrasing the initial question or constraints",
    "IN": "Logical Inference: Drawing a conclusion from prior steps",
    "EQ": "Equation Formulation: Setting up mathematical equations or logical relations",
    "CA": "Calculation: Arithmetic, symbolic execution, or direct step evaluation",
    "CO": "Conclusion: Final answer assertion or definitive terminal conclusion",
    "RF": "Reflection: Self-evaluation, sanity check, or questioning prior steps",
    "VE": "Verification: Back-substitution, constraint checks, or explicit validation",
    "DV": "Diversion: Off-topic tangent, redundant filler, or irrelevant discourse",
}

# Path to trained ensemble model
MODEL_PATH = PROJECT_ROOT / "models" / "production_ensemble" / "ensemble_scorer.pkl"

# Singleton model loader
_CACHED_SCORER: Optional[EnsembleScorer] = None

def get_scorer() -> EnsembleScorer:
    """Load or retrieve cached EnsembleScorer."""
    global _CACHED_SCORER
    if _CACHED_SCORER is not None:
        return _CACHED_SCORER

    if MODEL_PATH.exists():
        _CACHED_SCORER = EnsembleScorer.load(str(MODEL_PATH))
    else:
        # Fallback to training on balanced data if pre-saved model is missing
        from cot_hmm import EnsembleHMMTrainer
        import json
        data_path = PROJECT_ROOT / "data" / "v2" / "gsm8k_500_proper.json"
        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sequences = [d["tag_sequence"] for d in data if d.get("is_correct", False) and len(d.get("tag_sequence", [])) >= 2]
            trainer = EnsembleHMMTrainer(n_components=5, n_ensemble=12, n_iter=50, random_state=42)
            trainer.fit(sequences)
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            scorer = EnsembleScorer(trainer.ensemble)
            scorer.save(str(MODEL_PATH))
            _CACHED_SCORER = scorer
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH} and training data missing at {data_path}")
    return _CACHED_SCORER


def detect_structural_anomalies(tags: List[str]) -> List[Dict[str, Any]]:
    """Detect premature jumps, circular loops, and off-topic diversions."""
    anomalies = []
    
    # 1. Premature Conclusion Jump: CO without prior CA or EQ
    has_calc_or_eq = any(t in ("CA", "EQ") for t in tags[:-1])
    if "CO" in tags:
        co_idx = tags.index("CO")
        if co_idx < 2 and not has_calc_or_eq:
            anomalies.append({
                "type": "PREMATURE_CONCLUSION",
                "severity": "CRITICAL",
                "step_index": co_idx,
                "message": "Conclusion reached prematurely without sufficient intermediate derivation or calculation."
            })

    # 2. Circular / Oscillating Loop Detection (e.g. CA -> RF -> CA -> RF)
    for i in range(len(tags) - 3):
        if tags[i] == tags[i+2] and tags[i+1] == tags[i+3] and tags[i] != tags[i+1]:
            anomalies.append({
                "type": "CIRCULAR_REASONING_LOOP",
                "severity": "HIGH",
                "step_index": i,
                "pattern": f"{tags[i]} -> {tags[i+1]} -> {tags[i+2]} -> {tags[i+3]}",
                "message": "Reasoning is cycling between steps without progressing toward a solution."
            })
            break

    # 3. Off-Topic Diversions
    dv_indices = [idx for idx, t in enumerate(tags) if t == "DV"]
    if dv_indices:
        anomalies.append({
            "type": "IRRELEVANT_DIVERSION",
            "severity": "MEDIUM",
            "step_indices": dv_indices,
            "message": "Chain contains off-topic content or irrelevant tangents."
        })

    return anomalies


@server.tool(
    description=(
        "Score and verify a multi-step Chain-of-Thought reasoning trace. "
        "Segments the reasoning into semantic tags (PR, IN, EQ, CA, CO, RF, VE, DV), "
        "evaluates structural plausibility using an ensemble of 12 Hidden Markov Models, "
        "detects premature conclusion jumps and circular loops, and returns a PASS/REJECT decision."
    )
)
def score_reasoning_trace(trace: str) -> Dict[str, Any]:
    """
    Score a complete reasoning trace for structural plausibility.

    Args:
        trace: The raw multi-step reasoning text produced by an LLM or reasoning agent.

    Returns:
        Dict containing tags, plausibility score, anomalies, decision, and token savings estimate.
    """
    if not trace or not trace.strip():
        return {
            "error": "Empty reasoning trace provided",
            "decision": "REJECT",
            "plausibility_score": float("-inf")
        }

    steps = segment_and_clean(trace)
    if not steps:
        return {
            "error": "Could not segment reasoning into distinct steps",
            "decision": "REJECT",
            "plausibility_score": float("-inf")
        }

    tags = tag_steps(steps)
    scorer = get_scorer()
    score = scorer.score(tags)
    anomalies = detect_structural_anomalies(tags)

    # Decision rule based on empirical paper thresholds:
    # High score (>= -2.4): PASS
    # Borderline (-2.4 to -3.8): ESCALATE / SELF_CORRECT
    # Anomalous (< -3.8) or critical anomaly: REJECT
    has_critical_anomaly = any(a.get("severity") == "CRITICAL" for a in anomalies)
    has_circular_loop = any(a.get("type") == "CIRCULAR_REASONING_LOOP" for a in anomalies)

    if has_critical_anomaly or score < -3.8:
        decision = "REJECT"
        guidance = "Halt generation: reasoning exhibits severe structural failure or premature hallucination. Request replanning."
        compute_saved_pct = 45.0
    elif has_circular_loop or score < -2.4:
        decision = "ESCALATE"
        guidance = "Reasoning is borderline or oscillating. Encourage agent to verify calculation or break loop before concluding."
        compute_saved_pct = 20.0
    else:
        decision = "PASS"
        guidance = "Reasoning path is structurally valid and sound. Proceed to execution."
        compute_saved_pct = 0.0

    step_breakdown = [
        {"step_number": idx + 1, "tag": tag, "tag_name": TAG_DEFINITIONS.get(tag, "Unknown"), "text": step}
        for idx, (tag, step) in enumerate(zip(tags, steps))
    ]

    return {
        "decision": decision,
        "plausibility_score": round(score, 4),
        "tag_sequence": tags,
        "step_count": len(steps),
        "anomalies": anomalies,
        "guidance": guidance,
        "estimated_compute_saved_pct": compute_saved_pct,
        "step_breakdown": step_breakdown,
        "latency_profile": "148ms CPU inference (zero GPU required)"
    }


@server.tool(
    description=(
        "Decompose raw reasoning text into individual steps and classify each step into "
        "one of the 8 canonical CoT-HMM semantic tags."
    )
)
def tag_reasoning_steps(trace: str) -> Dict[str, Any]:
    """
    Decompose reasoning text and return semantic tag labels.

    Args:
        trace: Raw reasoning text.

    Returns:
        Dict with segmented steps, corresponding semantic tags, and vocabulary reference.
    """
    steps = segment_and_clean(trace)
    tags = tag_steps(steps)
    return {
        "step_count": len(steps),
        "tag_sequence": tags,
        "steps": [
            {"step": step, "tag": tag, "role": TAG_DEFINITIONS.get(tag, "Unknown")}
            for step, tag in zip(steps, tags)
        ],
        "tag_definitions": TAG_DEFINITIONS
    }


@server.tool(
    description=(
        "Online step-by-step verification during streaming reasoning. "
        "Evaluates whether appending candidate_step to previous_trace violates structural grammar."
    )
)
def verify_next_step(previous_trace: str, candidate_step: str) -> Dict[str, Any]:
    """
    Verify the transition from previous reasoning steps to a new proposed step.

    Args:
        previous_trace: Prior reasoning steps.
        candidate_step: The proposed next reasoning step.

    Returns:
        Dict with previous tags, next tag, delta plausibility score, and step verdict.
    """
    prev_steps = segment_and_clean(previous_trace) if previous_trace else []
    prev_tags = tag_steps(prev_steps) if prev_steps else []
    
    cand_steps = segment_and_clean(candidate_step)
    cand_tag = tag_steps(cand_steps)[0] if cand_steps else "IN"

    combined_tags = prev_tags + [cand_tag]
    scorer = get_scorer()
    
    prev_score = scorer.score(prev_tags) if prev_tags else -2.0
    combined_score = scorer.score(combined_tags)
    delta = combined_score - prev_score

    is_safe = delta > -1.5 and not (cand_tag == "CO" and not any(t in ("CA", "EQ") for t in prev_tags))

    return {
        "previous_tags": prev_tags,
        "candidate_tag": cand_tag,
        "candidate_role": TAG_DEFINITIONS.get(cand_tag, "Unknown"),
        "delta_score": round(delta, 4),
        "cumulative_plausibility": round(combined_score, 4),
        "step_verdict": "SAFE" if is_safe else "UNSAFE_TRANSITION",
        "warning": None if is_safe else "Step causes excessive structural drop or premature conclusion."
    }


@server.tool(
    description="Retrieve CoT-HMM architecture specs, benchmark metrics, and paper metadata."
)
def get_cot_hmm_specs() -> Dict[str, Any]:
    """Return model specifications, benchmark results, and citation info."""
    return {
        "name": "CoT-HMM Reasoning Guardrail",
        "paper_title": "CoT-HMM: Structural Plausibility Scoring of Chain-of-Thought Reasoning via Hidden Markov Models",
        "conference": "IEEE IDCIoT 2027",
        "paper_id": "IDCIoT-154",
        "authors": [
            "Gaurav Patil",
            "Suhail Shaikh",
            "Bhagyalakshmi V",
            "Parth Parmar",
            "Chirayu Marathe"
        ],
        "latency": "148 ms CPU (zero GPU needed)",
        "memory_footprint": "< 50 MB RAM",
        "ensemble_size": "M = 12 Hidden Markov Models",
        "vocabulary_size": 8,
        "benchmark_metrics": {
            "premature_jump_detection_auc": 0.990,
            "circular_loop_detection_auc": 0.761,
            "cross_architecture_transfer_auc": "0.981-0.992 (3 architectures)",
            "routing_pass_precision": "99.3%",
            "downstream_compute_token_savings": "42% to 47%"
        },
        "repository": "https://github.com/GauravPatil2515/Chain-Of-Taughts-HMM",
        "docs": "https://gauravpatil2515.github.io/Chain-Of-Taughts-HMM/"
    }


if __name__ == "__main__":
    import asyncio
    print("Starting CoT-HMM MCP Server over stdio...", file=sys.stderr)
    asyncio.run(server.run_stdio_async())
